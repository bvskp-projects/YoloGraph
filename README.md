# YoloGraph

Handwritten Images to Digital Format (Equivalently text)

## Objective

- Quick inference Time
- Learning object detection
- Patience Training :P

## Setup

- Install yamlu (in requirements.txt)
- Run yolog

```sh
pip install -r requirements.txt
./yolog -l=INFO preprocess
```

yolog, short for YoloGraph.

## Yolo Setup and Sample Model Training

To create the FCA/FCB yolo dataset, clone the yolov5 directory, and train a sample model see below. Yolov5 package dependencies were already installed with requirements.txt above.
```sh
python yolo_init.py
cd yolov5
python train.py --img 640 --epochs 3 --data ../yolo_dataset.yaml --weights ../yolov5s.pt
```

## Roadmap

- [x] Diagrams Dataset
- [x] DIDI Dataset

## Timing

- Diagrams dataset takes roughly on the order of 5 minutes from scratch and half a minute if data is already downloaded.
- DIDI dataset on the other hand, takes more than 10 minutes. Use the --skip-didi option to skip this processing.

## Design

First, preprocess data into the specified data format.

### Data Format

...

## How to Contribute

To add a new command, simply add a new subparser/action in yolog.py.
...
