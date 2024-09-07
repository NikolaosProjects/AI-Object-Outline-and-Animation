import json
import os

def delete_images_with_no_objects(json_file, images_folder):
    # Load the JSON file
    with open(json_file, 'r') as file:
        coco_data = json.load(file)

    # Track images that have objects
    images_with_objects = set()

    # Extract annotations from the COCO JSON data
    annotations = coco_data['annotations']
    for annotation in annotations:
        image_id = annotation['image_id']
        # Find the image filename from the COCO data
        for image in coco_data['images']:
            if image['id'] == image_id:
                image_filename = os.path.basename(image['file_name'])
                images_with_objects.add(image_filename)  # Mark this image as having objects
                break

    # Find and delete images without objects
    images_with_no_objects = [image['file_name'] for image in coco_data['images'] if os.path.basename(image['file_name']) not in images_with_objects]

    for image in images_with_no_objects:
        image_path = os.path.join(images_folder, image)
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted {image_path}")

    print("Images without objects have been deleted.")

# Example usage
json_file = "/home/nick/Coding/venv/2. AI Training Files/COCO2017/annotations/instances_val2017.json"  # JSON file
images_folder = "/home/nick/Coding/venv/2. AI Training Files/COCO2017/val2017"  # Folder containing images
delete_images_with_no_objects(json_file, images_folder)
