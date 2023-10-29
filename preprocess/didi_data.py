import json
import logging
import pydot
import matplotlib.patches as patches
import shutil

from pathlib import Path
from google.cloud import storage
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

# Settings
DATASETS = ["diagrams_wo_text", "diagrams"]
BUCKET_NAME = "digital_ink_diagram_data"
MAX_WORKERS = 8
PREPROC_DIR = 'didi_dataset'
SPLITS = {'test': 'test', 'train': 'train', 'valid': 'val'}

SHAPES = ['arrow', 'box', 'diamond', 'octagon', 'oval', 'parallelogram']


def preprocess_didi(cleanup=False):
  """
  Happens in the following steps:

  - Download dataset from google cloud storage
  - In particular, we only download ndjson and the xdot files
  - Setup folder structure data/{dataset}/{split}/{images,labels}
  - Output classes.txt for each dataset (set of class hardcoded to save time)
  - Parse xdot and ndjson
  - For each image in ndjson, stroke the ink in the images folder
  - Retrieve approximate bounding boxes from the corresponding xdot file
  - Print the shape and bbox into the labels folder

  Caveats:
  - Bounding boxes are very approximate
  - Some images have wildly incorrect labels
  """
  if cleanup:
    rmdata()
  create_dirs()
  download_didi()
  save_shapes()
  parse_drawings()


def PrepareDrawing():
  """ Setup the canvas for drawing """
  plt.clf()
  plt.axes().set_aspect("equal")
  plt.gca().yaxis.set_visible(False)
  plt.gca().xaxis.set_visible(False)
  plt.gca().axis('off')
  plt.gca().invert_yaxis()
    
    
def get_drawing_bb(ink):
  """ Get a rough bounding box for the drawing """
  minx = 99999
  miny = 99999
  maxx = 0
  maxy = 0

  for s in ink['drawing']:
    minx = min(minx, min(s[0]))
    maxx = max(maxx, max(s[0]))
    miny = min(miny, min(s[1]))
    maxy = max(maxy, max(s[1]))
  return (minx, miny, maxx - minx, maxy - miny)
    
    
def bbox_disjoint(bbox1, bbox2):
  """ Check if two bboxes are disjoint. Useful to filter out absurd bboxes. """
  x1, y1, w1, h1 = bbox1
  x2, y2, w2, h2 = bbox2
  
  # b1 left edge to right of b2 right edge
  # b1 right edge to left of b2 left edge
  # b1 bottom edge to top of b2 top edge
  # b1 top edge to bottom of b2 bottom edge
  return x1 > x2 + w2 or x1 + w1 < x2 or y1 > y2 + h2 or y1 + h1 < y2

    
def approx_bb(ddir, ink, print_bbox=False):
  """ Compute approximate bounding box for the image """
  # Compute scaling of the image.
  guide_width = ink["writing_guide"]["width"]
  guide_height = ink["writing_guide"]["height"]
  
  # Graph
  graph = pydot.graph_from_dot_file(str(Path(PREPROC_DIR, 'xdot', ink['label_id']).with_suffix('.xdot')))[0]
  bb = [float(coord) for coord in graph.get_bb().strip('"').split(',')]
  im_width, im_height = bb[2] - bb[0], bb[3] - bb[1]
  
  # Compute offsets
  scale=min(guide_width / im_width, guide_height / im_height)
  offset_x = (guide_width - scale * im_width) / 2
  offset_y = (guide_height - scale * im_height) / 2
  
  # Collect bboxes for each shape and arrow
  bboxes = []
  src_bbox = get_drawing_bb(ink)
  inv_shapes = {shape: i for i, shape in enumerate(SHAPES)}
  
  for node in graph.get_nodes():
    shape = node.get_shape()
    if shape is None:
      continue

    pos = [float(coord) for coord in node.get_pos().strip('"').replace('\\\n', '').split(',')]
    cw = offset_x + scale * pos[0]
    ch = offset_y + scale * (im_height - pos[1])
    bbscale = 75
    bbw, bbh = float(node.get_width()) * bbscale * scale, float(
       node.get_height()) * bbscale * scale
    minw, minh = cw - bbw/2, ch - bbh/2
    # Check if bbox includes some stroke (only an approximation)
    if bbox_disjoint(src_bbox, (minw, minh, bbw, bbh)):
      return False
    bboxes.append([inv_shapes[shape], minw, minh, bbw, bbh])
    if print_bbox:
      rectangle = patches.Rectangle((cw - bbw/2, ch - bbh/2), bbw, bbh, 
                        linewidth=1, edgecolor='b', facecolor='none')
      plt.gca().add_patch(rectangle)
  
  for edge in graph.get_edges():
    # Edge
    pos = edge.get_pos()
    if pos is None:
      continue

    pos = [[float(feat) for feat in coord.split(',')] for coord in pos.strip('"').replace('\\\n', '').lstrip('e,').split(' ')]
    ws, hs = [coord[0] for coord in pos], [coord[1] for coord in pos]
    minw, maxw = offset_x + scale * min(ws), offset_x + scale * max(ws)
    minh, maxh = offset_y + scale * (im_height - max(hs)), offset_y + scale * (im_height - min(hs))
    # Ensure minimum width, weight (10 * scale)
    if maxw - minw < 10 * scale:
      minw -= 5 * scale
      maxw += 5 * scale
    if maxh - minh < 10 * scale:
      minh -= 5 * scale
      maxh += 5 * scale
    bbw, bbh = maxw - minw, maxh - minh
    # Filter out absolutely absurd drawings
    if bbox_disjoint(src_bbox, (minw, minh, bbw, bbh)):
      return False
    bboxes.append([inv_shapes['arrow'], minw, minh, bbw, bbh])
    if print_bbox:
      rectangle = patches.Rectangle((minw, minh), bbw, bbh, 
                        linewidth=1, edgecolor='r', facecolor='none')
      plt.gca().add_patch(rectangle)
          
  # Save these bboxes to a file
  with open(Path(ddir, 'labels', ink['label_id']).with_suffix('.txt'), 'w') as file:
    for bbox in bboxes:
      file.write(' '.join([str(round(label, 1)) for label in bbox]) + '\n')
          
  return True

    
def display_strokes(ink):
  """ Display strokes in the image """
  for s in ink["drawing"]:
    plt.plot(s[0], [y for y in s[1]], color="black")

        
def display_ink(ddir, ink):
  """ Draw image, overlay strokes, approximate bbox. """
  PrepareDrawing()
  display_strokes(ink)
  scheck = approx_bb(ddir, ink, print_bbox=False)
  if not scheck:
    # Sanity Check Failed.
    # Do not print. Ignore this figure.
    return
  canvas = FigureCanvas(plt.gcf())
  canvas.print_figure(str(Path(ddir, 'images', ink['label_id']).with_suffix('.png')), bbox_inches='tight')


def parse_drawings():
  """ Parse the ndjson and xdot files """
  for dataset in DATASETS:
    ddir = data_dir(dataset)
    
    # Collect only the relevant drawings
    # We pick the first drawing for each label_id
    drawings = []
    ink_set = set()
    with open(json_file(dataset), 'r') as file:
      for line in file:
        ink = json.loads(line)
        label_id = ink['label_id']
        if label_id in ink_set:
          continue
        ink_set.add(label_id)
        
        drawings.append(ink)
    
    # Now iterate over drawings to
    # - print strokes
    # - approx bb
    for ink in tqdm(drawings):
      split = SPLITS[ink['split']]
      display_ink(f'{ddir}/{split}', ink)


def json_file(dataset):
  """ json file corresponding to dataset """
  return f'{PREPROC_DIR}/{dataset}_20200131.ndjson'


def data_dir(dataset):
  """ data directory corresponding to dataset """
  return f'data/didi_{dataset}'


def create_dirs():
  """ Dir structure for post processed data """
  logging.info("Setting up dataset directory structure...")
  for dataset in DATASETS:
    ddir = data_dir(dataset)
    for split in SPLITS.values():
      Path(ddir, split, 'images').mkdir(parents=True, exist_ok=True)
      Path(ddir, split, 'labels').mkdir(parents=True, exist_ok=True)


def download_blob(blob):
  """ For multithreading """
  blob.download_to_filename(str(Path(PREPROC_DIR, blob.name)))
    

def download_didi():
  """ Download ndjson and xdot files from google cloud storage """
  # Data already downloaded, skip ...
  if Path(PREPROC_DIR, 'xdot').exists():
    logging.warn("Data already downloaded, skipping...")
    return
  
  # Google cloud bucket
  logging.info("Downloading data from Google Cloud Storage ...")
  Path(PREPROC_DIR, 'xdot').mkdir(parents=True, exist_ok=True)
  
  # Google Storage Client
  client = storage.Client.create_anonymous_client()
  bucket = client.bucket(bucket_name=BUCKET_NAME)
  
  # Download ndjson and xdot files
  validext = lambda bname: bname.endswith('.ndjson') or bname.endswith('.xdot')
  blobs = [blob for blob in bucket.list_blobs() if validext(blob.name)]
  
  thread_map(download_blob, blobs, max_workers=MAX_WORKERS)


def save_shapes():
  """ Print the hardcoded list to classes.txt """
  for dataset in DATASETS:
    with open(Path(data_dir(dataset), 'classes.txt'), 'w') as file:
        for line in SHAPES:
          file.write(f'{line}\n')


def collect_shapes():
  """
  Unused at the moment. Used to generate hardcoded SHAPES.
  """
  # Arrows are not explicitly mentioned. They are simply edges.
  shapes = set(['arrow'])

  for _, _, files in os.walk(Path(PREPROC_DIR, 'xdot')):
    for file in files:
      graph = pydot.graph_from_dot_file(Path(PREPROC_DIR, 'xdot', file))[0]
      for node in graph.get_nodes():
        shape = node.get_shape()
        if shape is not None and shape not in shapes:
          shapes.add(shape)

  return shapes

def rmdata():
  data_dir = Path(PREPROC_DIR)
  if data_dir.exists():
    logging.info(f"Removing {data_dir} ...")
    shutil.rmtree(data_dir)
