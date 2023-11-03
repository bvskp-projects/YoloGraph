from pathlib import Path
import subprocess
import argparse
import shutil
import os
import random

TEXT_DIR = "deep-text-recognition-benchmark"
GITHUB_URL = 'https://github.com/clovaai/deep-text-recognition-benchmark.git'
PRETRAINED_URL = 'https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW'
TEXT_ROOT = Path(TEXT_DIR)

DATA_DIR = "text_dataset"
DATA_ROOT = Path(DATA_DIR)
LMDB_DIR = "text_dataset_lmdb"
LMDB_ROOT = Path(LMDB_DIR)


def parse_args():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser(prog='Text Initialization',
                    description='Set up the deep-text-recognition-benchmark directory and potentially train some models')
    return parser.parse_args()

def clone_text():
    """ Clone the text directory from GitHub """
    print("Downloading deep-text-recognition-benchmark from GitHub ...")
    # Check GitHub repo exists
    if TEXT_ROOT.exists():
      print("Skipping download since deep-text-recognition-benchmark exists")
      return
    
    subprocess.run(["git", "clone", GITHUB_URL])


def setup_data():
    """ Set up text formatted data """
    print("Setting up data...")
    if DATA_ROOT.exists():
        print("Skipping setting up data as root already exists")
        return

    os.mkdir("text_dataset")
    os.mkdir("text_dataset/test")
    os.mkdir("text_dataset/train/")
    new_gt_file_path_train = "text_dataset/gt_train.txt"
    new_gt_file_path_test = "text_dataset/gt_test.txt"


    old_data_dir = "text_data"
    old_gt_file_path = "text_data/gt.txt"

    p = 0.1 # proportion of images to be put in test set

    with open(new_gt_file_path_train, 'w') as train_f:
        with open(new_gt_file_path_test, 'w') as test_f:
            with open(old_gt_file_path, 'r') as f:
                for line in f:
                    line_list = line.strip().split(" ")
                    img_path = line_list[0]
                    label = ' '.join(line_list[1:])
                    img = img_path.split("/")[-1]
                    if random.uniform(0, 1) <= p:
                        # move to test set
                        shutil.move(os.path.join(old_data_dir, img_path), os.path.join("text_dataset/test", img))  
                        test_f.write(f"{os.path.join('test', img)}\t{label}\n")
                    else:
                        # move to train set
                        shutil.move(os.path.join(old_data_dir, img_path), os.path.join("text_dataset/train", img))
                        train_f.write(f"{os.path.join('train', img)}\t{label}\n")

    print("Finished setting up the data.")

def setup_lmdb_datasets():
    """Set up lmdb formatted text data for model training/eval."""
    print("Setting up lmdb dirs...")

    if LMDB_ROOT.exists():
        print("Skipping setting up lmdb data as root already exists...")
        return
    
    os.mkdir("text_dataset_lmdb")
    os.mkdir("text_dataset_lmdb/train")
    os.mkdir("text_dataset_lmdb/test")

    print("Finished setting up up lmdb data dirs NOW CREATE LMDB DATASETS BY RUNNING:")
    print("1. $ cd deep-text-recognition-benchmark")
    print("2. $ python create_lmdb_dataset.py --inputPath ../text_dataset/ --gtFile ../text_dataset/gt_test.txt --outputPath ../text_dataset_lmdb/test/")
    print("3. $ python create_lmdb_dataset.py --inputPath ../text_dataset/ --gtFile ../text_dataset/gt_train.txt --outputPath ../text_dataset_lmdb/train/")
    print("\n NOTE: if you run into lmdb errors saying you don't have enough disk space go to deep-text-recognition-benchmark/create_lmdb_dataset.py and change line 38 to have map_size=1073741824")




    

if __name__ == "__main__":
    args = parse_args()
    setup_data()
    clone_text()
    setup_lmdb_datasets()
    