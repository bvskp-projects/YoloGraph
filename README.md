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

To create the FCA/FCB yolo dataset, clone the yolov5 directory, and train a model see below. Yolov5 package dependencies were already installed with requirements.txt above. Download the pre-trained yolov5s.pt model from their github.
```sh
python yolo_init.py
cd yolov5
python train.py --img 640 --epochs 100 --data ../yolo_dataset.yaml --weights ../pretrained_models/yolov5s.pt
```

## Text Setup and Sample Model Training

Download the text_data.zip file from the google drive, extract it to text_data/ directory. Download the pre-trained TRBA-case sensitive model. Then run the code below to create the text_dataset_lmdb, and clone the deep-text-recognition-benchmark directory. 
```sh
python text_init.py
cd deep-text-recognition-benchmark
python train.py --train_data ../train_dataset_lmdb/train/ --valid_data ../train_dataset_lmdb/test/ --saved_model ../pretrained_models/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth --FT --select_data / --batch_ratio 1 --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --workers 0 --num_iter 300 --valInterval 5 --sensitive
```

To see some sample results run this command while in ```deep-text-recognition/``` directory with ```TRBA_best_accuracy.pth``` as recently trained model and sample images in ```demo_image/``` directory:
```sh
python demo.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn --image_folder demo_image/ --saved_model ../models/TRBA_best_accuracy.pth --sensitive
```

## Using the trained models to decode diagrams

Look at **ExampleDiagramDecoding.ipynb** for how to do this. The code used in this notebook come from the text directory as well as the decode_diagrams.py file.

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
