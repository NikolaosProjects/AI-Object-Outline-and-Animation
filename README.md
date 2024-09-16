<h1 align="center"><b>Animating Outlines: AI & Fourier Series</b></h1>

<h3 align="center"><b>CAT</b></h3>
<div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
  <img src="https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Results/1.%20Cat/Cat%20Image.png" alt="Cat Image" width="49%" height="320px" style="object-fit: cover;">
  <img src="https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Results/1.%20Cat/Cat%20Outline%20Animation.gif" alt="Cat Animation Gif" width="49%" height="320px" style="object-fit: cover;">
</div>

<h1 align="center"></h1>

<h3 align="center"><b>CAR</b></h3>
<div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
  <img src="https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Results/2.%20Car/Car%20Image.png" alt="Car Image" width="49%" height="285px" style="object-fit: cover;">
  <img src="https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Results/2.%20Car/Car%20Outline%20Animation.gif" alt="Car Animation Gif" width="49%" height="285px" style="object-fit: cover;">
</div>

<h1 align="center"></h1>

<h3 align="center"><b>PLANE</b></h3>
<div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
  <img src="https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Results/3.%20Plane/Plane%20Image.png" alt="Plane Image" width="49%" height="335px" style="object-fit: cover;">
  <img src="https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Results/3.%20Plane/Plane%20Outline%20Animation.gif" alt="Plane Animation Gif" width="49%" height="335px" style="object-fit: cover;">
</div>

<h1 align="center"></h1>

<h3 align="center"><b>BICYCLE</b></h3>
<div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
  <img src="https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Results/4.%20Bicycle/Bicycle%20Image.png" alt="Bicycle Image" width="49%" height="295px" style="object-fit: cover;">
  <img src="https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Results/4.%20Bicycle/Bicycle%20Outline%20Animation.gif" alt="Bicycle Animation Gif" width="49%" height="295px" style="object-fit: cover;">
</div>

<h1 align="center"></h1>

<h1 align="center"><b>Project Description</b></h1>

For this project, I trained an image segmentation model to accurately identify and trace the outlines of Cats, Cars, Planes, and Bicycles. I converted these borders into points in the complex plane, and analyzed them using Fourier Analysis. I used the fourier coefficients and their frequencies as rotating vectors, and used their rotation to trace the outline they originated from.

<h1 align="center"></h1>

<h3 align="left"><b>Artificial Intelligence Model</b></h3>

I used the untrained YOLOv8s-seg model from Ultralytics (https://docs.ultralytics.com/tasks/segment), which is an Artificial Intelligence image detection and segmentation model. When provided with an image, it can identify objects from up to 80 different categories. Additionally, it can detect the location of these objects in the image, and identify their exact borders.

<h1 align="center"></h1>

<h3 align="left"><b>Model Training</b></h3>

For my project I wanted to provide my own training to the model, with the goal of identifying Cats, Cars, Planes and Bicycles. I used Train and Validation images from the COCO2017 dataset (https://cocodataset.org/#download). Each of these images' name is a unique ID and the dataset is accompanied by "annotations". These are .json files which link each image's unique ID with a list of all the objects in that image, as well as the outlines of these objects as sets of (x, y) points (coordinates). One "annotations" file can contain the IDs and properties of thousands of images. 

I used the SAMA-COCO annotations (https://www.sama.com/sama-coco-dataset), as they provide object outlines with higher detail compared to the stock COCO2017 annotations. This dataset's annotations contains several .json files. I started by merging all the "training" immages' annotations into one .json file, and all the "validation" images' annotations into one .json file.

<h1 align="center"></h1>
<details>
  <summary>Click to expand Python code</summary>
  
  ```python
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
      train_files = glob('path to train annotations folder')
      merge_json_files(train_files, 'instances_train2017.json')
  
      # Merge validation JSON files
      val_files = glob('path to val annotations folder')
      merge_json_files(val_files, 'instances_val2017.json')
```
</details>

<h1 align="center"></h1>

The COCO2017 dataset consists of around 181,000 images. For my training purposes, I only wanted the images that contained Cats, Cars, Planes, and Bicycles. I went through the .json "annotations" file, and extracted the image IDs and object details for only the pictures from the COCO dataset that contained the desired objects. I used that information to create new filtered "annotations" files (containing only the image IDs and details for the desired categories), and duplicated the images of interest into a new folder. That is how I created my custom training dataset's images and annotations.

<h1 align="center"></h1>

<details>
  <summary>Click to expand Python code</summary>
  ```python
  # THIS FOLLOWS AFTER DEFINING THE PATH VARIABLES (SEE SCRIPT IN REPOSITORY)
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
  ```
</details>

<h1 align="center"></h1>

The YOLOv8s-seg model cannot read through the .json "annotations" file that contains all the image IDs and their attributes. It needs a unique .txt text file for each image, that is named exactly the same as the image ID, and contains the details of all objects contained in that specific image. Using a script by z00bean (https://github.com/z00bean/coco2yolo-seg), i went through my filtered dataset "annotations" and created a text file for each of my filtered images, with each of the text files containing all the attributes of its contained objects. This is z00bean's script:

<h1 align="center"></h1>

```python
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
```
<h1 align="center"></h1>
During the training procedure, the model passes through the large set of train images, examining each picture and its corresponding .txt file. After a full pass, its performance is evaluated by comparing its own inference on objects of the validation images, with the exact attributes of these objects as defines in the .txt annotation files. This cycle constitutes one "epoch". I trained my model on 100 epochs.

<h1 align="center"></h1>

```python
from ultralytics import YOLO

# Load YOLOv8 segmentation model
model = YOLO('yolov8s_seg.yaml')  # Load untrained model (performs segmentation but is completely untrained)

# Train the model using your dataset and specific classes
model.train(
    data='/home/nick/Coding/venv/2. AI Training Files/YOLOv8s-seg/config.yaml',  # Dataset YAML
    epochs=100,      # Number of epochs
    imgsz=1024,       # Image size
    batch=10,        # Batch size
    workers=4,       # Data loaders
    device=0,        # GPU (set to 'cpu' if needed)
    name='path to trained weights folder',  # Save path of new training
    resume=True,  # Specify the checkpoint file
    cache=0,
    lr0=0.005,
    lrf=0.0005
)
```
<h1 align="center"></h1>

<h3 align="left"><b>Using the Trained YOLOv8s-seg</b></h3>

After training my model, I uploaded the weights (model's training knowledge) on my gihub so it can be easily accessible by anyone who wants to download my code and try this for themselves. I set options within my script to use the weights directly from my github link, and to utilize the system's GPU if it's available as it offers better performance compared to the CPU. 

The user can set the main variable at the top of the script to the image they wish to analyze (options are: 1 for Cat, 2 for Car, 3 for Plane, 4 for Bicycle).

The script automatically loads the appropriate picture using a link to my github (no need to download the picture), sets all the graph and animation parameters for the specific picture, and presents the selected image to the user.

The image's dimensions are changed to 640x640. The resized image is then converted into a format of (channel, height, width), its values changed to float for higher accuracy, and then changed to appear as a batch of size 1. After setting these parameters, we turn the image into a tensor, and divide by 255 in order to have the range of RGB values describing the image, set between 0 and 1. These are very important parameters that the model needs in order to be able to analyze the picture.

The model processes the image, and the processed image is returned to the user with the detected object highlighted in red color. 

At this point, using CV2 (image and color processing module), and the output of the model, the script identifies the coordinates of the object's boundary, and stores these coordinates as a list of complex (imaginary) points in the form of (x + iy).

<h1 align="center"></h1>

<h3 align="left"><b>Fourier Analysis</b></h3>

At this stage, our outline is converted into a discrete set of data in the complex plane, that can be analyzed using Fourier Series. using FFT (Fast Fourier Transform) algorithm, the script extracts the outline's Fourier coefficients (complex vectors (x, iy)) as well as their associated frequencies. It then pairs them correctly, and places them symmetrically around f = 0hz.

The outline of the object in the form of complex points, and the distribution of fourier coefficients' magnitude given their frequency are returned to the user.

<h1 align="center"></h1>

<h3 align="left"><b>Outline Animation</b></h3>

Since the fourier coefficients are vectors in the complex plane, and have a specific frequency associated with them, we can make them rotate. if we pair them tip to tail, and make each vector rotate about the point of the previous vector, tracing the path of the last vector's tip results in tracing the object's outline, given a sufficiencly large number fourier coefficients (number of rotating vectors), to ensure accuracy.

Using the python module Manim, it is relatively easy to achieve this. We use the fourier coefficients' coordinates to define a vector corresponding to each coefficient, and then define its angular velocity (rate of rotation) using omega = 2*pi*f. Manim creates an animation by updating a dt variable, where dt is the amount of time in seconds between frames (60fps means dt = 1/60s). To make the vectors rotate, we need to update their position after time dt. to do that, we make them rotate by an angle theta, defined by their frequency. Mathematically, a vector roating at frequency f, after time duration dt, will roate by an angle: theta = omega*dt => theta = 2*pi*f*dt.

After each frame, we update the positions of each vector, by rotating it with its corresponding angle theta. To make the animations go faster, we multiplied the frequency of each vector by 500 (it litearlly makes every signle vector rotate 500 times faster).
