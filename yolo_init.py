from pathlib import Path
import subprocess
import argparse
import shutil
import os
import json
import collections
import random

YOLO_DIR = "yolov5"
GITHUB_URL = 'https://github.com/ultralytics/yolov5.git'
PRETRAINED_URL = 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt'
YOLO_ROOT = Path(YOLO_DIR)

DATA_DIR = "yolo_dataset"
DATA_ROOT = Path(DATA_DIR)


def parse_args():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(prog='Yolo Initialization',
                    description='Set up the yolov5 directory and potentially train some models')
    return parser.parse_args()

def clone_yolo():
    """ Clone the yolov5 directory from GitHub """
    print("Downloading yolov5 from GitHub ...")
    # Check GitHub repo exists
    if YOLO_ROOT.exists():
      print("Skipping download since yolov5 exists")
      return
    
    subprocess.run(["git", "clone", GITHUB_URL])

def write_labels_txt_file(labels, dest_file):
    """ Given label list, create labels.txt at destination file """
    with open(dest_file, 'w') as f:
        for label in labels:
            f.write(label + '\n')
            
def move_all_imgs(src_dir, dest_dir, keep=4):
    """ Move all images in src directory to dest directory. By default, copy the first four images so src dir is not completely empty.  """
    ctr = 0
    for img in list(filter(lambda x: "jpg" in x or "png" in x,  os.listdir(src_dir))):
        if ctr < keep:
            shutil.copy(os.path.join(src_dir, img), os.path.join(dest_dir, img))
            ctr += 1
        else:
            shutil.move(os.path.join(src_dir, img), os.path.join(dest_dir, img))

def parse_coco_json_to_yolo_with_cocoreader(coco_dir, dest_dir, labels, label_map=None, train=True, doPrint=False):
    """ Parse the COCO format json file in coco_dir to create yolo format txt files in dest_dir
        - labels is the list of labels to be used
        - If train, use train split of COCO dataset """
    label_set = set(labels)
    
    scan_reader = CocoReader(Path(coco_dir))

    if train:
        ann_imgs = scan_reader.parse_split("train")
    else:
        ann_imgs = scan_reader.parse_split("test")

    for image in ann_imgs:
        img_name = image.filename
        img_width, img_height = image.size
        with open(os.path.join(dest_dir, img_name[:-4] + '.txt'), 'w') as f:
            for annotation in image.annotations:
                if (label_map and label_map[annotation.category] not in label_set) or (not label_map and annotation.category not in label_set):
                    print(f"WARNING: There is an annotation not in the label set/map with label {annotation.text}.")
                    continue
                
                xmin = annotation.bb.l
                ymin = annotation.bb.t
                xmax = annotation.bb.r
                ymax = annotation.bb.b
                
                xcenter = ((xmin + xmax) / 2) / img_width
                ycenter = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                if label_map:
                    f.write(f"{labels.index(label_map[annotation.category])} {xcenter} {ycenter} {width} {height}\n")
                    if doPrint:
                        print(img_name, labels.index(label_map[annotation.category]), xcenter, ycenter, width, height)
                else:
                    f.write(f"{labels.index(annotation.category)} {xcenter} {ycenter} {width} {height}\n")
                    if doPrint:
                        print(img_name, labels.index(annotation.category), xcenter, ycenter, width, height)


def setup_data():
    """ Set up yolo formatted data """
    print("Setting up data...")
    if DATA_ROOT.exists():
        print("Skipping setting up data as root already exists")
        return

    os.mkdir("yolo_dataset")
    os.mkdir("yolo_dataset/images")
    os.mkdir("yolo_dataset/labels/")

    data_dirs = {
        'fa': "handwritten-diagram-datasets/datasets/fa", # finite automata - FINAL STATES ARE NOT LABELED CORRECLTY (DOUBLE CIRCLES) FOR OUR PURPOSES: SKIP
        'fca': "handwritten-diagram-datasets/datasets/fca", # online flowchart: INCLUDE
        'fcb': "handwritten-diagram-datasets/datasets/fcb", # offline flowchart: INCLUDE
        'fcb_scan': "handwritten-diagram-datasets/datasets/fcb_scan", # DUPLICATE OF fcb: SKIP
        'didi': "handwritten-diagram-datasets/datasets/didi"} # SKIP FOR NOW because complicated...
    labels_dict = {
        'final': ['circle', 'rectangle', 'parallelogram', 'diamond', 'arrow', 'text'], # labels we aim to use in the yolo model
        'general': ['connection', 'data', 'decision', 'process', 'terminator', 'text', 'arrow'] # labels used in the fca/fcb datasets
    }
    label_map = { # what each COCO label should become in the YOLO labels
        'connection': 'circle',
        'data': 'parallelogram',
        'decision': 'diamond',
        'process': 'rectangle',
        'terminator': 'circle',
        'text': 'text',
        'arrow': 'arrow'
    }

    # Create correct label files for each image file for each fca/fcb train/test directory
    parse_coco_json_to_yolo_with_cocoreader(data_dirs['fcb'], "yolo_dataset/labels", labels_dict['final'], label_map)
    parse_coco_json_to_yolo_with_cocoreader(data_dirs['fcb'], "yolo_dataset/labels", labels_dict['final'], label_map, train=False)
    parse_coco_json_to_yolo_with_cocoreader(data_dirs['fca'], "yolo_dataset/labels", labels_dict['final'], label_map)
    parse_coco_json_to_yolo_with_cocoreader(data_dirs['fca'], "yolo_dataset/labels", labels_dict['final'], label_map, train=False)

    # Move all image files to the yolo image directories
    move_all_imgs(os.path.join(data_dirs["fcb"], "train"), "yolo_dataset/images")
    move_all_imgs(os.path.join(data_dirs["fcb"], "test"), "yolo_dataset/images")
    move_all_imgs(os.path.join(data_dirs["fca"], "train"), "yolo_dataset/images")
    move_all_imgs(os.path.join(data_dirs["fca"], "test"), "yolo_dataset/images")

    # Create labels.txt file
    write_labels_txt_file(labels_dict['final'], "yolo_dataset/labels.txt")

    # Random train/test split
    os.mkdir("yolo_dataset/train")
    os.mkdir("yolo_dataset/test")
    os.mkdir("yolo_dataset/train/images")
    os.mkdir("yolo_dataset/train/labels")
    os.mkdir("yolo_dataset/test/images")
    os.mkdir("yolo_dataset/test/labels")

    p = 0.1 # proportion of images to be put in test set

    img_list = os.listdir("yolo_dataset/images")
    for img in img_list:
        if random.uniform(0, 1) <= p:
            # move to test set
            shutil.move(os.path.join("yolo_dataset/images", img), os.path.join("yolo_dataset/test/images", img))  
            shutil.move(os.path.join("yolo_dataset/labels", img[:-4] + '.txt'), os.path.join("yolo_dataset/test/labels", img[:-4] + '.txt'))  
        else:
            # move to train set
            shutil.move(os.path.join("yolo_dataset/images", img), os.path.join("yolo_dataset/train/images", img))  
            shutil.move(os.path.join("yolo_dataset/labels", img[:-4] + '.txt'), os.path.join("yolo_dataset/train/labels", img[:-4] + '.txt')) 

    os.rmdir("yolo_dataset/images")
    os.rmdir("yolo_dataset/labels")
    print("Finished setting up the data.")
    

if __name__ == "__main__":
    args = parse_args()
    setup_data()
    clone_yolo()
    