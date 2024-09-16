"""
import json
import os
import shutil

# Define paths
coco_train_annotations_path = '/home/nick/Coding/venv/2. AI Training Files/YOLOv8s-seg/Datasets/Full/annotations/instances_train2017.json'
coco_val_annotations_path = '/home/nick/Coding/venv/2. AI Training Files/YOLOv8s-seg/Datasets/Full/annotations/instances_val2017.json'
coco_train_images_dir = '/home/nick/Coding/venv/2. AI Training Files/YOLOv8s-seg/Datasets/Full/images/train/'
coco_val_images_dir = '/home/nick/Coding/venv/2. AI Training Files/YOLOv8s-seg/Datasets/Full/images/val/'

# Paths for filtered outputs
filtered_train_annotations_path = '/home/nick/Coding/venv/2. AI Training Files/YOLOv8s-seg/Datasets/Filtered/annotations/instances_train2017.json'
filtered_val_annotations_path = '/home/nick/Coding/venv/2. AI Training Files/YOLOv8s-seg/Datasets/Filtered/annotations/instances_val2017.json'
filtered_train_images_dir = '/home/nick/Coding/venv/2. AI Training Files/YOLOv8s-seg/Datasets/Filtered/images/train/'
filtered_val_images_dir = '/home/nick/Coding/venv/2. AI Training Files/YOLOv8s-seg/Datasets/Filtered/images/val/'
YOLO_train_annotations = "/home/nick/Coding/venv/2. AI Training Files/YOLOv8s-seg/Datasets/Filtered/labels/train/"
YOLO_val_annotations = "/home/nick/Coding/venv/2. AI Training Files/YOLOv8s-seg/Datasets/Filtered/labels/val/"

# Define categories of interest
desired_categories = {"cat": 17, "car": 3, "airplane": 5, "bicycle": 2}

# Function to delete a directory if it exists
def delete_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)

# Delete destination folders to start from scratch
delete_directory(filtered_train_annotations_path.rsplit('/', 1)[0])
delete_directory(filtered_val_annotations_path.rsplit('/', 1)[0])
delete_directory(filtered_train_images_dir)
delete_directory(filtered_val_images_dir)
delete_directory(YOLO_train_annotations)
delete_directory(YOLO_val_annotations)

# Recreate the directories
os.makedirs(filtered_train_annotations_path.rsplit('/', 1)[0], exist_ok=True)
os.makedirs(filtered_val_annotations_path.rsplit('/', 1)[0], exist_ok=True)
os.makedirs(filtered_train_images_dir, exist_ok=True)
os.makedirs(filtered_val_images_dir, exist_ok=True)
os.makedirs(YOLO_train_annotations, exist_ok=True)
os.makedirs(YOLO_val_annotations, exist_ok=True)

# Helper function to filter annotations and images
def filter_coco_data(annotations_path, images_dir, filtered_annotations_path, filtered_images_dir, desired_category_names):
    # Load COCO annotations
    with open(annotations_path, 'r') as f:
        data = json.load(f)

    # Get the list of desired category IDs
    desired_category_ids = [cat['id'] for cat in data['categories'] if cat['name'] in desired_category_names]

    # Filter annotations
    filtered_annotations = {
        'images': [],
        'annotations': [],
        'categories': [cat for cat in data['categories'] if cat['name'] in desired_category_names]
    }
    filtered_image_ids = set()

    for ann in data['annotations']:
        if ann['category_id'] in desired_category_ids:
            filtered_annotations['annotations'].append(ann)
            filtered_image_ids.add(ann['image_id'])

    for img in data['images']:
        if img['id'] in filtered_image_ids:
            filtered_annotations['images'].append(img)

    # Save filtered annotations
    with open(filtered_annotations_path, 'w') as f:
        json.dump(filtered_annotations, f)

    # Copy filtered images
    for img in filtered_annotations['images']:
        src_img_path = os.path.join(images_dir, img['file_name'])
        dest_img_path = os.path.join(filtered_images_dir, img['file_name'])
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dest_img_path)
        else:
            print(f"Warning: {src_img_path} does not exist and will not be copied.")

# Apply the filter for train and validation datasets
filter_coco_data(
    coco_train_annotations_path,
    coco_train_images_dir,
    filtered_train_annotations_path,
    filtered_train_images_dir,
    desired_categories
)

filter_coco_data(
    coco_val_annotations_path,
    coco_val_images_dir,
    filtered_val_annotations_path,
    filtered_val_images_dir,
    desired_categories
)

# Mapping of original category IDs to new category IDs
category_mappings = {
    17: 0,  # cat
    3: 1,   # car
    5: 2,   # airplane
    2: 3    # bicycle
}

def update_category_ids(json_path, category_mappings):
    # Load COCO annotations
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Print existing category IDs for debugging
    existing_ids = set(ann['category_id'] for ann in data['annotations'])
    print(f"Existing category IDs: {existing_ids}")
    print(f"Category mappings: {category_mappings}")

    # Update category IDs in annotations
    for annotation in data['annotations']:
        old_id = annotation['category_id']
        if old_id in category_mappings:
            annotation['category_id'] = category_mappings[old_id]
        else:
            print(f"Warning: Category ID {old_id} not found in category_mappings.")
    
    # Update category IDs in categories
    for category in data['categories']:
        old_id = category['id']
        if old_id in category_mappings:
            category['id'] = category_mappings[old_id]
        else:
            print(f"Warning: Category ID {old_id} not found in category_mappings.")

    # Save updated annotations
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Updated annotations saved to {json_path}")

# Apply the updates
update_category_ids(filtered_train_annotations_path, category_mappings)
update_category_ids(filtered_val_annotations_path, category_mappings)


def convert_coco_to_yolo_segmentation(json_file, folder_name="labels"):
    folder_name = folder_name
    # Load the JSON file
    with open(json_file, 'r') as file:
        coco_data = json.load(file)

    # Create a "labels" folder to store YOLO segmentation annotations
    output_folder = os.path.join(os.path.dirname(json_file), folder_name)
    os.makedirs(output_folder, exist_ok=True)

    # Extract annotations from the COCO JSON data
    annotations = coco_data['annotations']
    for annotation in annotations:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        segmentation = annotation['segmentation']
        bbox = annotation['bbox']

        # Find the image filename from the COCO data
        for image in coco_data['images']:
            if image['id'] == image_id:
                image_filename = os.path.basename(image['file_name'])
                image_filename = os.path.splitext(image_filename)[0] # Removing the extension. (In our case, it is the .jpg or .png part.)
                image_width = image['width']
                image_height = image['height']
                break

        # Calculate the normalized center coordinates and width/height
        x_center = (bbox[0] + bbox[2] / 2) / image_width
        y_center = (bbox[1] + bbox[3] / 2) / image_height
        bbox_width = bbox[2] / image_width
        bbox_height = bbox[3] / image_height

        # Check if segmentation is a list of lists (polygon) or RLE
        if isinstance(segmentation, list):
            # Convert COCO segmentation to YOLO segmentation format
            yolo_segmentation = [f"{(x) / image_width:.5f} {(y) / image_height:.5f}" for x, y in zip(segmentation[0][::2], segmentation[0][1::2])]
            yolo_segmentation = ' '.join(yolo_segmentation)
        else:
            continue

        # Generate the YOLO segmentation annotation line
        yolo_annotation = f"{category_id} {yolo_segmentation}"

        # Save the YOLO segmentation annotation in a file
        output_filename = os.path.join(output_folder, f"{image_filename}.txt")
        with open(output_filename, 'a+') as file:
            file.write(yolo_annotation + '\n')

    print("Conversion completed")

#change train annotations to yolo format
convert_coco_to_yolo_segmentation(filtered_train_annotations_path, YOLO_train_annotations)

#change val annotaions to yolo format
convert_coco_to_yolo_segmentation(filtered_val_annotations_path, YOLO_val_annotations)
"""

from ultralytics import YOLO

# Load YOLOv8 segmentation model
model = YOLO('/home/nick/Coding/venv/2. AI Training Files/YOLOv8s-seg/Trained Model/yolov8s_seg_trained/weights/last.pt')  # Load untrained model (performs segmentation but is completely untrained)

# Train the model using your dataset and specific classes
model.train(
    data='/home/nick/Coding/venv/2. AI Training Files/YOLOv8s-seg/config.yaml',  # Dataset YAML
    epochs=350,      # Number of epochs
    imgsz=1024,       # Image size
    batch=10,        # Batch size
    workers=4,       # Data loaders
    device=0,        # GPU (set to 'cpu' if needed)
    name='/home/nick/Coding/venv/2. AI Training Files/YOLOv8s-seg/Trained Model/yolov8s_seg_trained',  # Save path of new training
    resume=True,  # Specify the checkpoint file
    cache=0,
    lr0=0.005,
    lrf=0.0005
)