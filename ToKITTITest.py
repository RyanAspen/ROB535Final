import os
import numpy as np
import copy
from operator import itemgetter 
from PIL import Image
import shutil

rng1 = np.random.default_rng()
data_dir = "/home/jupyter"
test_path = os.path.join(data_dir, "Test")

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
    

all_camera = parse_calibration(test_path, "0")

# Test set frames
int_frames = sorted([int(i) for i in list(all_camera.keys())])

if not os.path.exists("testing/calib"):
    os.makedirs("testing/calib")
if not os.path.exists("testing/image_2"):
    os.makedirs("testing/image_2")

cur_file = None
test_set = []

image_inds = []
for test_ind in int_frames:
        # Image file
        if test_ind < 423:
            image_path = os.path.join(test_path, "Camera", "rgb_" + str(test_ind).zfill(5)+".jpg")
        else:
            image_path = os.path.join(test_path, "CameraExtraCredit", "rgb_" + str(test_ind).zfill(5)+".jpg")
        
        new_image_path = os.path.join("testing/image_2/", str(test_ind).zfill(6) + ".png")
        
        # Some images are skipped
        try:
            im = Image.open(image_path)
            image_inds.append(test_ind)
        except:
            continue
        im.save(new_image_path)

        # Calibration file
        calib_path = os.path.join("testing/calib", str(test_ind).zfill(6)+".txt")
        calib_info = all_camera[str(test_ind)]
        write_calib(calib_path, calib_info)

if not os.path.exists("ImageSets"):
    os.makedirs("ImageSets")

# Train split
file_path = os.path.join("ImageSets/test.txt")
if os.path.exists(file_path):
    os.remove(file_path)
cur_file = open(file_path, "w")
for ind in image_inds:
    line = str(ind).zfill(6)
    cur_file.write(line + "\n")
cur_file.close()
