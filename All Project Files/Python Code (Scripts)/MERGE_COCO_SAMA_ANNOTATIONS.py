import json
import os
from glob import glob

def merge_json_files(input_files, output_file):
    combined_data = {"images": [], "annotations": [], "categories": []}
    for file_name in input_files:
        with open(file_name, 'r') as f:
            data = json.load(f)
            combined_data['images'].extend(data.get('images', []))
            combined_data['annotations'].extend(data.get('annotations', []))
            if not combined_data['categories']:
                combined_data['categories'] = data.get('categories', [])

    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=4)


if __name__ == "__main__":
    # Merge training JSON files
    train_files = glob('/home/nick/Coding/venv/2. AI Training Files/YOLOv8s-seg/Datasets/Full/annotations/sama-coco-train/*.json')
    merge_json_files(train_files, 'instances_train2017.json')

    # Merge validation JSON files
    val_files = glob('/home/nick/Coding/venv/2. AI Training Files/YOLOv8s-seg/Datasets/Full/annotations/sama-coco-val/*.json')
    merge_json_files(val_files, 'instances_val2017.json')
