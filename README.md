# YoloGraph: Offline Hand-Drawn Diagram Conversion

Handwritten Diagram Images to Digital Format. Please see ``YoloGraph_Final_Report.pdf`` for more details.

## Objective

- Accurate diagram node detection
- Accurate text recognition
- Accurate arrow key point extrapolation
- Quick inference Time

## Setup

- Create an environment with python version 3.10
- Install packages in requirements.txt
- Run yolog

```sh
pip install -r requirements.txt
./yolog -l=INFO preprocess
```

yolog, short for YoloGraph.

## Node Detection Model Setup and Training

To create the FCA/FCB node detection dataset, clone the yolov5 directory, and train a model see below. Yolov5 package dependencies were already installed with requirements.txt above. Download the pre-trained yolov5s.pt model from their github.
```sh
python yolo_init.py
cd yolov5
python train.py --img 640 --epochs 100 --data ../yolo_dataset.yaml --weights ../pretrained_models/yolov5s.pt
```

## Text Recognition Model Setup and Training

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

Look at **ExampleDiagramDecoding.ipynb** for how to do this. The code used in this notebook come from the text directory as well as the ```decode_diagrams.py``` file. To download the pretrained model files go to the google drive or click here: [node detection model](https://drive.google.com/file/d/1ufcdRJSt2qbtIDRsJA9z57-7nLS99a9O/view?usp=drive_link), [text recognition model](https://drive.google.com/file/d/1I9GpfRgAOmtQCqcgYDYLzQWD-EMl9Q1N/view?usp=drive_link). 

## References

For a list of references please look at our report. 
