import os
import numpy as np
import copy
from operator import itemgetter 
from PIL import Image
import shutil

rng1 = np.random.default_rng()
data_dir = "/home/jupyter"
train_path = os.path.join(data_dir, "Train")
val_path = os.path.join(data_dir, "Val")

# Moving val set to be a subset of training data
new_dest = os.path.join(train_path, "03")
if not os.path.exists(new_dest):
    try:
        shutil.copytree(val_path, new_dest)
    except:
        raise
else:
    print("Path", new_dest, "already exists")

# Read
kept_order = {"frame": 0, "truncation_ratio": 1, "occupancy_ratio": 2, "alpha": 3,
             "left": 4, "top": 5, "right": 6, "bottom": 7, "height": 8, "width": 9, "length": 10,
             "camera_space_X": 11, "camera_space_Y": 12, "camera_space_Z": 13, "rotation_camera_space_y": 14,
             "confidence": 15}

def get_pose(fpath_pose, all_ordered):
    pose_keys = {}
    file = open(fpath_pose, "r")
    i = 0
    for line in file.readlines():
        line = line.strip()
        broken_line = line.split(" ")
        # Header file
        if i == 0:
            for j in range(len(broken_line)):
                word = broken_line[j]
                pose_keys[word] = j
            x = set(kept_order.keys())
            y = set(pose_keys.keys())
            look_at = x.intersection(y)
            i += 1
            continue
        # Only keep camera 0
        if str(broken_line[1]) != str(0):
            continue
        # Keep words
        for word in look_at:
            j = pose_keys[word]
            all_ordered[word].append(broken_line[j])
            # Add confidence
            all_ordered["confidence"].append(str(rng1.random()))
    file.close()
    return all_ordered

def get_bbox(fpath_pixels, all_ordered):
    bbox_keys = {}
    old_frames = copy.deepcopy(all_ordered["frame"])
    all_ordered["frame"] = []

    file = open(fpath_pixels, "r")
    i = 0
    for line in file.readlines():
        line = line.strip()
        broken_line = line.split(" ")
        # Header file
        if i == 0:
            for j in range(len(broken_line)):
                word = broken_line[j]
                bbox_keys[word] = j
            x = set(kept_order.keys())
            y = set(bbox_keys.keys())
            look_at = x.intersection(y)
            i += 1
            continue
        # Only keep camera 0
        if str(broken_line[1]) != str(0):
            continue
        # Keep words
        for word in look_at:
            j = bbox_keys[word]
            if word =="occupancy_ratio":
                value = float(broken_line[j])
                if value > 0.5:
                    all_ordered[word].append(str(0))
                elif value > 0.2:
                    all_ordered[word].append(str(1))
                elif value > 0.05:
                    all_ordered[word].append(str(2))
                else:
                    all_ordered[word].append(str(3))
            else:
                all_ordered[word].append(broken_line[j])
    file.close()
    assert(old_frames == all_ordered["frame"])
    return all_ordered

def get_all(dir_path):
    all_ordered = {"frame": [], "truncation_ratio": [], "occupancy_ratio": [], "alpha": [],
             "left": [], "top": [], "right": [], "bottom": [], "height": [], "width": [], "length": [],
             "camera_space_X": [], "camera_space_Y": [], "camera_space_Z": [], "rotation_camera_space_y": [],
             "confidence": []}
    fpath_pose = os.path.join(dir_path, "pose.txt")
    fpath_pixels = os.path.join(dir_path, "bbox.txt")
    all_ordered = get_pose(fpath_pose, all_ordered)
    all_ordered = get_bbox(fpath_pixels, all_ordered)
    return all_ordered

def to_line(words):
    new_line = "Car "
    for word in words[:-1]:
        new_line += word + " "
    new_line += words[-1] + "\n"
    return new_line

ordered_keys = ["truncation_ratio", "occupancy_ratio", "alpha",
             "left", "top", "right", "bottom", "height", "width", "length",
             "camera_space_X", "camera_space_Y", "camera_space_Z", "rotation_camera_space_y"]

# Calibration
def parse_calibration(folder, cameraID):
    frame_lines = {}
    file = open(os.path.join(folder, "intrinsic.txt"), "r")
    i = 0
    for line in file.readlines():
        line = line.strip()
        broken_line = line.split(" ")
        if i != 0:
            if broken_line[1] == cameraID:
                frame_id = broken_line[0]
                new_info = broken_line[2:]
                frame_lines[frame_id] = new_info
        i += 1
    file.close()
    return frame_lines

def calib_text(calib_info):
    P = calib_info[0] + " 0.0 " + calib_info[2] + " 0.0 0.0 " + calib_info[1] + " " + calib_info[3] + " 0.0 0.0 0.0 1.0 0.0\n"
    lines = []
    for i in range(4):
        new_line = "P" + str(i) + ": " + P
        lines.append(new_line)
    lines.append("R0_rect: 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0\n")
    identity = "1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0\n"
    lines.append("Tr_velo_to_cam: " + identity)
    lines.append("Tr_imu_to_velo: " + identity)
    return lines

def write_calib(fpath, calib_info):
    if os.path.exists(fpath):
        os.remove(fpath)
    calib_lines = calib_text(calib_info)
    cur_file = open(fpath, "w")
    for new_line in calib_lines:
        cur_file.write(new_line)
    cur_file.close()
    

all_camera = {}
for train_dir in ["00", "01", "02", "03"]:
    frame_lines = parse_calibration(os.path.join(train_path, "00"), "0")
    all_camera[train_dir] = frame_lines

if not os.path.exists("training/calib"):
    os.makedirs("training/calib")
if not os.path.exists("training/image_2"):
    os.makedirs("training/image_2")
if not os.path.exists("training/label_2"):
    os.makedirs("training/label_2")

total_frames = -1
cur_file = None
train_set = []
val_set = []
for train_dir in ["00", "01", "02", "03"]:
    dir_path = os.path.join(train_path, train_dir)
    all_ordered = get_all(dir_path)
    frames = all_ordered["frame"]
    prev_frame = -1
    for i in range(len(frames)):
        
        # Check if new frame
        if frames[i] != prev_frame:
            # Close old file
            if cur_file is not None:
                cur_file.close()
                
            # New frame
            prev_frame = frames[i]
            total_frames += 1
            
            # New path for labels
            new_fpath = os.path.join("training/label_2", str(total_frames).zfill(6) + ".txt")
            if os.path.exists(new_fpath):
                os.remove(new_fpath)
            cur_file = open(new_fpath, "w")
            
            # Record the file number
            if train_dir == "03":
                val_set.append(str(total_frames).zfill(6))
            else:
                train_set.append(str(total_frames).zfill(6))
            
            # Image file
            image_path = os.path.join(train_path, train_dir, "Camera", "rgb_" + frames[i].zfill(5)+".jpg")
            new_image_path = os.path.join("training/image_2/", str(total_frames).zfill(6) + ".png")
            im = Image.open(image_path)
            im.save(new_image_path)
            
            # Calibration file
            calib_path = os.path.join("training/calib", str(total_frames).zfill(6)+".txt")
            calib_info = all_camera[train_dir][frames[i]]
            write_calib(calib_path, calib_info)
        new_line = [all_ordered[word][i] for word in ordered_keys]
        new_line = to_line(new_line)
        cur_file.write(new_line)
# Close file
cur_file.close()

trainval_set = train_set + val_set

if not os.path.exists("ImageSets"):
    os.makedirs("ImageSets")

# Train split
file_path = os.path.join("ImageSets/train.txt")
if os.path.exists(file_path):
    os.remove(file_path)
cur_file = open(file_path, "w")
for line in train_set:
    cur_file.write(line + "\n")
cur_file.close()

# Val split
file_path = os.path.join("ImageSets/val.txt")
if os.path.exists(file_path):
    os.remove(file_path)
cur_file = open(file_path, "w")
for line in val_set:
    cur_file.write(line + "\n")
cur_file.close()

# TrainVal split
file_path = os.path.join("ImageSets/trainval.txt")
if os.path.exists(file_path):
    os.remove(file_path)
cur_file = open(file_path, "w")
for line in trainval_set:
    cur_file.write(line + "\n")
cur_file.close()
