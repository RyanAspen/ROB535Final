import os
import sys
import torch
import torch.optim as optim
import numpy as np
import csv

from tqdm.auto import tqdm
from typing import Dict, List
from yacs.config import CfgNode
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model import MonoConDetector
from engine.base_engine import BaseEngine
from dataset.monocon_dataset import MonoConDataset
from solver import CyclicScheduler

from utils.visualizer import Visualizer
from utils.decorators import decorator_timer
from utils.engine_utils import progress_to_string_bar, move_data_device, reduce_loss_dict, tprint


class MonoconEngine(BaseEngine):
    def __init__(self, cfg: CfgNode, **kwargs):    
        super().__init__(cfg, **kwargs)

        
    def build_model(self):
        detector = MonoConDetector(
            num_dla_layers=self.cfg.MODEL.BACKBONE.NUM_LAYERS,
            pretrained_backbone=self.cfg.MODEL.BACKBONE.IMAGENET_PRETRAINED)
        return detector.to(self.current_device)
    
    
    def build_solver(self):
        assert (self.model is not None)
        assert (self.train_loader is not None)
        
        optimizer = optim.AdamW(
            params=self.model.parameters(),
            lr=self.cfg.SOLVER.OPTIM.LR,
            weight_decay=self.cfg.SOLVER.OPTIM.WEIGHT_DECAY,
            betas=(0.95, 0.99))
        
        scheduler = None
        if self.cfg.SOLVER.SCHEDULER.ENABLE:
            total_steps = (len(self.train_loader) * self.cfg.SOLVER.OPTIM.NUM_EPOCHS)
            scheduler = CyclicScheduler(
                optimizer,
                total_steps=total_steps,
                target_lr_ratio=(10, 1E-04),
                target_momentum_ratio=(0.85 / 0.95, 1.0),
                period_up=0.4)
            
        return optimizer, scheduler
    
    
    def build_loader(self, is_train: bool = True):
        dataset = MonoConDataset(
            base_root=self.cfg.DATA.ROOT,
            split=self.cfg.DATA.TRAIN_SPLIT if is_train else self.cfg.DATA.TEST_SPLIT,
            max_objs=self.cfg.MODEL.HEAD.MAX_OBJS,
            filter_configs={k.lower(): v for k, v in dict(self.cfg.DATA.FILTER).items()})
        
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.DATA.BATCH_SIZE,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            shuffle=True if is_train else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)
        return dataset, loader


    @decorator_timer
    def train_one_epoch(self) -> float:
        epoch_losses = []
        for batch_idx, data_dict in enumerate(self.train_loader):
            
            self.optimizer.zero_grad()
            
            # Forward
            data_dict = move_data_device(data_dict, self.current_device)
            _, loss_dict = self.model(data_dict)
            total_loss = reduce_loss_dict(loss_dict)
            total_loss.backward()
            
            # Save Losses
            step_loss = total_loss.detach().item()
            epoch_losses.append(step_loss)
            self.entire_losses.append(step_loss)
            
            # Clip Gradient (Option)
            if self.cfg.SOLVER.CLIP_GRAD.ENABLE:
                clip_args = {k.lower(): v for k, v in dict(self.cfg.SOLVER.CLIP_GRAD).items()
                             if k not in ['ENABLE']}
                clip_grad_norm_(self.model.parameters(), **clip_args)
            
            # Step
            self.optimizer.step()
            if (self.scheduler is not None):
                self.scheduler.step()
            
            # Update and Log
            if (self.global_iters % self.log_period == 0):
                one_epoch_steps = len(self.train_loader)
                prog_bar = progress_to_string_bar((batch_idx + 1), one_epoch_steps, bins=20)
                recent_loss = sum(self.entire_losses[-100:]) / len(self.entire_losses[-100:])
                print(f"| Progress {prog_bar} | LR {self.current_lr:.6f} | Loss {total_loss.item():8.4f} ({recent_loss:8.4f}) |")
                
                self._update_dict_to_writer(loss_dict, tag='loss')
                
            self._iter_update()
        self._epoch_update()

        # Return Average Loss
        epoch_loss = (sum(epoch_losses) / len(epoch_losses))
        return epoch_loss
    
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        
        cvt_flag = False
        if self.model.training:
            self.model.eval()
            cvt_flag = True
            tprint("Model is converted to eval mode.")
            
        eval_container = {
            'img_bbox': [],
            'img_bbox2d': []}
        
        for test_data in tqdm(self.test_loader, desc="Collecting Results..."):
            test_data = move_data_device(test_data, self.current_device)
            eval_results = self.model.batch_eval(test_data)
            
            for field in ['img_bbox', 'img_bbox2d']:
                eval_container[field].extend(eval_results[field])
        
        eval_dict = self.test_dataset.evaluate(eval_container,
                                               eval_classes=['Pedestrian', 'Cyclist', 'Car'],
                                               verbose=True)
        
        if cvt_flag:
            self.model.train()
            tprint("Model is converted to train mode.")
        return eval_dict

    @torch.no_grad()
    def infer(self):

        def kitti_3d_to_file(annos: Dict[str, List[Dict]], img_metas: Dict, folder: str, single_file: bool = False):
            """Write the detection results to a file each frame or altogether.
            annos: a listdetection results of a mini batch, output from MonoConDetector.batch_eval().
            img_metas: a dictionary containing the meta information of the mini batch, output from MonoConDataset.
            single_file: whether to write the detection results of all frames to a single file.
            """
            annos = annos['img_bbox']
            print("Length of annos = ", len(annos))
            print("Length of img_metas = ", len(img_metas))
            if single_file:
                if not os.path.exists(os.path.dirname(folder)):
                    os.makedirs(os.path.dirname(folder))
            else:
                if not os.path.exists(folder):
                    os.makedirs(folder)
            
            if single_file:
                ### Open the file for the whole dataset
                predict_txt = folder
                f = open(predict_txt, 'a', newline='')
                w = csv.writer(f, delimiter=' ', lineterminator='\n')

            ### Iterate over all frames
            for i, anno in enumerate(annos):
                ### Find the sample index
                sample_idx = img_metas['sample_idx'][i]

                if not single_file:
                    ### Open a file for the current frame
                    predict_txt = os.path.join(folder, f'{sample_idx:06d}.txt')
                    f = open(predict_txt, 'w', newline='')
                    w = csv.writer(f, delimiter=' ', lineterminator='\n')

                num_det = len(anno['sample_idx'])

                ### Skip if there is no detection
                if num_det == 0:
                    if not single_file:
                        f.close()
                    continue

                for det_idx in range(num_det):    
                    ### Get the detection information and form a row
                    name = anno['name'][det_idx]
                    truncated = anno['truncated'][det_idx]
                    occluded = anno['occluded'][det_idx]
                    alpha = anno['alpha'][det_idx]
                    bbox = anno['bbox'][det_idx]
                    dimensions = anno['dimensions'][det_idx]
                    location = anno['location'][det_idx]
                    rotation_y = anno['rotation_y'][det_idx]
                    score = anno['score'][det_idx]

                    # if name not in CLASSES:
                    if name != 'Car':
                        continue

                    print("Car detected with bbox = ", bbox, "and score = ", score)

                    ### Write the detection to the file
                    if single_file:
                        w.writerow([sample_idx, name, truncated, occluded, alpha, *bbox, *dimensions, *location, rotation_y, score])
                    else:
                        w.writerow([name, truncated, occluded, alpha, *bbox, *dimensions, *location, rotation_y, score])

                if not single_file:
                    f.close()

            if single_file:
                f.close()
            return

        cvt_flag = False
        if self.model.training:
            self.model.eval()
            cvt_flag = True
            tprint("Model is converted to eval mode. (Infer)")
            
            for test_data in tqdm(self.test_loader, desc="Collecting Results..."):
                test_data = move_data_device(test_data, self.current_device)
                annos = self.model.batch_eval(test_data) #Dict[str, Any]
                img_metas = test_data['img_metas']
                kitti_3d_to_file(annos, img_metas, "output")
        
        if cvt_flag:
            self.model.train()
            tprint("Model is converted to train mode.")
        return
    
    @torch.no_grad()
    def visualize(self, 
                  output_dir: str, 
                  draw_items: List[str] = ['bev', '2d', '3d']):
        
        cvt_flag = False
        if self.model.training:
            self.model.eval()
            cvt_flag = True
            tprint("Model is converted to eval mode.")
        
        vis_container = []
        scale_hw = None
        
        for test_data in tqdm(self.test_loader, desc="Collecting Results..."):
            test_data = move_data_device(test_data, self.current_device)
            
            if (scale_hw is None) and test_data['img_metas'].get('scale_hw', False):
                scale_hw = test_data['img_metas']['scale_hw'][0]    
            
            vis_results = self.model.batch_eval(test_data, get_vis_format=True)
            vis_container.extend(vis_results)
            
        if scale_hw is not None:
            tprint(f"Visualization will be progressed using scale factor {scale_hw}.")
        visualizer = Visualizer(self.test_dataset, vis_format=vis_container, scale_hw=scale_hw)
        draw_item_to_func = {
            '2d': 'plot_bboxes_2d',
            '3d': 'plot_bboxes_3d',
            'bev': 'plot_bev'}
        
        for draw_item in draw_items:
            save_dir = os.path.join(output_dir, draw_item)
            os.makedirs(save_dir, exist_ok=True)
            
            for idx in tqdm(range(len(self.test_dataset)), desc=f"Visualizing '{draw_item.upper()}'..."):
                draw_func = getattr(visualizer, draw_item_to_func[draw_item])
                
                ori_filename = os.path.basename(self.test_dataset[idx]['img_metas']['image_path'])
                draw_func(idx, save_path=os.path.join(save_dir, ori_filename))
                
        if cvt_flag:
            self.model.train()
            tprint("Model is converted to train mode.")
