# Instructions to generate predictions

1) Download project dataset and unzip it
2) Update ToKITTITrain.py and ToKITTITest.py so data_dir is set to the directory where the project dataset was unzipped to
3) Run "python ToKITTITest.py"
4) Update exps/config.yaml so DATA/ROOT is set to the path with the KITTI data
5) Run "python test.py --config_file exps/config.yaml --checkpoint_file /exps/checkpoints/{CHECKPOINT_FILE} --evaluate"
6) Run "python merger.py --folder_path output/ --save_path {DIRECTORY TO SAVE PREDICTIONS TO}"
7) The generated file has all of the predictions for the test set. All predictions after frame 499 are for the extra credit test set.
