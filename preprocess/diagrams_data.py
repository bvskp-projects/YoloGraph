import logging
import subprocess
import shutil

from pathlib import Path
from yamlu.coco_read import CocoReader

# Settings
PREPROC_DIR = 'handwritten-diagram-datasets/datasets'
POSTPROC_DIR = 'data'
DATASETS = ['fa', 'fca', 'fcb', 'fcb_scan', 'hdBPMN-icdar2021']
SPLITS = ['test', 'train', 'val']
GITHUB_URL = 'https://github.com/bernhardschaefer/handwritten-diagram-datasets.git'

DATASETS_ROOT = Path(PREPROC_DIR)

# Categories for each dataset
CLASSES = {
    'fa': ['text', 'final state', 'arrow', 'state'],
    'fca': ['terminator', 'arrow', 'data', 'process', 'text', 'decision', 'connection'],
    'fcb': ['terminator', 'arrow', 'data', 'process', 'text', 'decision', 'connection'],
    'fcb_scan': ['terminator', 'arrow', 'data', 'process', 'text', 'decision', 'connection'],
    'hdBPMN-icdar2021': ['parallelGateway', 'timerEvent', 'lane', 'messageFlow', 'task', 'dataStore', 'dataObject', 'exclusiveGateway', 'pool', 'sequenceFlow', 'subProcess', 'dataAssociation', 'event', 'eventBasedGateway', 'messageEvent']
}


def preprocess_diagrams(cleanup=False):
  """
  Happens in the following steps:

  - Download the dataset from GitHub
  - Create the required directories
  - Create classes.txt for each dataset containing categories
  - Parse json files to retrieve bounding boxes
  - Copy input images from GitHub to the dataset directory
  """
  if cleanup:
    rmdata()
  download_diagrams()
  create_dirs()
  save_classes()
  parse_bb()
  copy_images()


def parse_bb():
  """ Parse json files to retrieve bounding boxes """
  logging.info("Processing bounding box labels...")
  # For fast index lookup given category.
  inv_classes = {dataset: {
     cat: i for i, cat in enumerate(categories)
    } for dataset, categories in CLASSES.items()}

  for dataset in DATASETS:
    reader = CocoReader(DATASETS_ROOT / dataset)
    # Output bb for each split in separate folder.
    for split in splits(dataset):
        ann_imgs = reader.parse_split(split)
        # One label file per image.
        for ai in ann_imgs:
            label_path = Path(POSTPROC_DIR, dataset, split, 'labels', ai.img_id
                              ).with_suffix('.txt')
            with open(str(label_path), 'w') as file:
                for ann in ai.annotations:
                    ## Extract class, x, y, w, h <-> bounding box
                    category, bb = ann.category, ann.bb
                    if category is not None and bb is not None:
                        cls = inv_classes[dataset][category]
                        x, y, w, h = bb.bb_coco
                        # Output format
                        file.write(f'{cls} {x} {y} {w} {h}\n')


def copy_images():
  """ Must happen after everything since we destructively modify the repo """
  logging.info("Copying images ...")
  for dataset in DATASETS:
    for split in splits(dataset):
        src_dir = Path(PREPROC_DIR, dataset, split)
        dst_dir = Path(POSTPROC_DIR, dataset, split, 'images')
        if dst_dir.exists():
          logging.warn(f"Overwriting {dst_dir} ...")
          shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)


def save_classes():
  """ Write these categories to file data -> {dataset} -> classes.txt """
  logging.info("Saving classes.txt ...")
  for dataset, categories in CLASSES.items():
      # Common classes.txt across all splits.
      with open(str(Path(POSTPROC_DIR, dataset, 'classes.txt')), 'w') as file:
          for cat in categories:
              file.write(f'{cat}\n')


def download_diagrams():
  """ Download the dataset from GitHub """
  # Check GitHub repo exists
  if DATASETS_ROOT.exists():
    logging.warn("Skipping download since handwritten-diagram-datasets exists")
    return

  logging.info("Downloading handwritten-diagram-datasets from GitHub ...")
  subprocess.run(["git", "clone", GITHUB_URL, "--depth=1"])


def create_dirs():
  """ Create the required directories """
  for dataset in DATASETS:
    for split in splits(dataset):
      Path(POSTPROC_DIR, dataset, split, 'labels').mkdir(parents=True, exist_ok=True)


def splits(dataset):
  """ Unfortunate but fca does not have a val dataset """
  return ['test', 'train', 'val'] if  dataset != 'fca' else ['test', 'train']


def collect_categories():
  """
  Not used in current code but used to generate an exhaustive list of
  categories. CLASSES is generated using this code.
  """
  shapes = dict()

  for dataset in DATASETS:
      categories = set()
      reader = CocoReader(DATASETS_ROOT / dataset)
      for split in splits(dataset):
          ann_imgs = reader.parse_split(split)
          for ai in ann_imgs:
              categories.update(ai.categories)
          
      shapes[dataset] = list(categories)

  print(shapes)

def rmdata():
  data_dir = Path(PREPROC_DIR).parent
  if data_dir.exists():
    logging.info(f"Removing {data_dir} ...")
    shutil.rmtree(data_dir)
