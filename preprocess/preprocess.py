import logging
import shutil
import time

from .diagrams_data import preprocess_diagrams
from .didi_data import preprocess_didi
from pathlib import Path

def preprocess(cleanup=False, skip_didi=False):
  """
  TODO: Add support for preprocessing other datasets, ex. DIDI
  """
  if cleanup:
    data_dir = Path('data')
    if data_dir.exists():
      logging.info('Removing data directory ...')
      shutil.rmtree(data_dir)
  start = time.time()
  preprocess_diagrams(cleanup=cleanup)
  logging.info(f'preprocess_diagrams took {time.time() - start:.0f} seconds')
  if not skip_didi:
    start = time.time()
    preprocess_didi(cleanup=cleanup)
    logging.info(f'preprocess_didi took {time.time() - start:.0f} seconds')
