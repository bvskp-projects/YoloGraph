from decode_diagrams import create_yolo_model, create_text_model, decode_diagram_image

from collections import defaultdict
from pathlib import Path
from PIL import Image

import os


## Class Mapping from dataset specific classes to yolo trained classes

fa_classmap = {
    "text": "text",
    "final state": "circle",
    "arrow": "arrow",
    "state": "circle",
}

fca_classmap = {
    'connection': 'circle',
    'data': 'parallelogram',
    'decision': 'diamond',
    'process': 'rectangle',
    'terminator': 'circle',
    'text': 'text',
    'arrow': 'arrow',
}

fcb_classmap = {
    'connection': 'circle',
    'data': 'parallelogram',
    'decision': 'diamond',
    'process': 'rectangle',
    'terminator': 'circle',
    'text': 'text',
    'arrow': 'arrow',
}

fcb_scan_classmap = {
    'connection': 'circle',
    'data': 'parallelogram',
    'decision': 'diamond',
    'process': 'rectangle',
    'terminator': 'circle',
    'text': 'text',
    'arrow': 'arrow',
}

didi_classmap = {
    "arrow": "arrow",
    "box": "rectangle",
    "diamond": "diamond",
    "octagon": "circle",
    "oval": "circle",
    "parallelogram": "parallelogram",
}

classmap = {
    "fa": fa_classmap,
    "fca": fca_classmap,
    "fcb": fcb_classmap,
    "fcb_scan": fcb_scan_classmap,
    "didi_diagrams_wo_text": didi_classmap,
}


datasets = ["fa", "fca", "fcb", "fcb_scan", "didi_diagrams_wo_text"]


def run_tests():
    """
    Run all tests across all datasets to compute similarity score.
    """
    # Create yolo model
    yolo_model_path = "models/Yolov5s_best.pt"
    yolo_model = create_yolo_model(yolo_model_path)

    # Create text model
    text_model_path = "models/TRBA_best_accuracy.pth"
    converter, text_model, AlignCollate_demo = create_text_model(text_model_path)

    for dataset in datasets:
        # Computes the fraction of test images that are correctly drawn
        # We use a simple metric that summarizes the graph structure
        print(f"Running similarity test for {dataset=}")
        images = extract_images(dataset)
        outcomes = decode_diagram_image(images, yolo_model, text_model, converter, AlignCollate_demo, object_thresh=0.5)
        similarity_score = compute_similarity(dataset, outcomes)
        print(f"Similarity score for {dataset=} is {similarity_score}")


def extract_images(dataset):
    """ Extracts images in the test directory """
    # Extract images from the test directory
    img_dir = get_test_dir(dataset, "images")
    image_path_list = os.listdir(img_dir)
    image_path_list = sorted(image_path_list) # Not guaranteed to list them in order
    images = [Image.open(os.path.join(img_dir, image_path)) for image_path in image_path_list]
    if dataset in {"fcb_scan", "didi_diagrams_wo_text"}:
        # These images are not grayscale and the models expect grayscale images
        images = [img.convert("L") for img in images]
    return images


def compute_num_shapes(outcome):
    """
    Given outcome of the yolo model and the text model as decoded by decode_diagram_image
    Returns the number of instances of each shape
    """
    num_shapes = defaultdict(int)
    for tup in outcome:
        tup_class = tup[0]
        if tup_class != "text":
            num_shapes[tup_class] += 1
    return num_shapes


def get_test_dir(dataset, idir):
    """ Returns test image or label directory """
    return str(Path("data", dataset, "test", idir))


def compute_similarity(dataset, outcomes):
    """
    Computes similarity given yolo compute_outcomes
    Assumes that the images are listed in the same order as the labels
    """
    correct = 0
    label_dir = get_test_dir(dataset, "labels")
    label_path_list = os.listdir(label_dir)
    label_path_list = sorted(label_path_list) # Not guaranteed to list them in order
    label_path_list = [Path(label_dir, label_path) for label_path in label_path_list]
    assert len(outcomes) == len(label_path_list)
    for outcome, label_path in zip(outcomes, label_path_list):
        pred_num_shapes = compute_num_shapes(outcome)
        target_num_shapes = parse_num_shapes(dataset, label_path)
        if pred_num_shapes == target_num_shapes:
            correct += 1
    print(f"Got {correct} images correct out of {len(outcomes)}")
    return correct / len(outcomes)


def parse_num_shapes(dataset, label_path):
    """ Computes the number of shapes in the labels file. """
    # Parse the list of classes from classes.txt
    # This maps the class index to class name
    # Required to parse 
    classes = []
    with open(Path("data", dataset, "classes.txt"), "r") as f:
        for line in f:
            classes.append(line.strip())
    assert len(classes) == len(classmap[dataset])

    # Extract shapes to compute the number of shapes
    num_shapes = defaultdict(int)
    with open(label_path, "r") as f:
        for line in f:
            classname = classes[int(line.split()[0])]
            num_shapes[classname] += 1

    # Convert classnames from dataset specific to yolo trained classes
    # Multiple dataset specific classes may map to trained classes
    final_num_shapes = defaultdict(int)
    for shape, count in num_shapes.items():
        classname = classmap[dataset][shape]
        if classname != "text":
            final_num_shapes[classname] += count

    return final_num_shapes
