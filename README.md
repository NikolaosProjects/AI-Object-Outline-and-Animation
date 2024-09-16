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

<h1 align="center"><b>PROJECT DESCRIPTION</b></h1>

For this project, I trained an image segmentation model to accurately identify and trace the outlines of Cats, Cars, Planes, and Bicycles. I converted these borders into points in the complex plane, and analyzed them using Fourier Analysis. I used the fourier coefficients and their frequencies as rotating vectors, and used their rotation to trace the outline they originated from.

<h1 align="center"></h1>

<h3 align="center"><b>🔺 ARTIFICIAL INTELLIGENCE MODEL🔺</b></h3>

I used the untrained YOLOv8s-seg model from Ultralytics (https://docs.ultralytics.com/tasks/segment), which is an Artificial Intelligence image detection and segmentation model. When provided with an image, it can identify objects from up to 80 different categories. Additionally, it can detect the location of these objects in the image, and identify their exact borders.

<h1 align="center"></h1>

<h3 align="center"><b>🔺 MODEL TRAINING🔺</b></h3>

For my project I wanted to provide my own training to the model, with the goal of identifying Cats, Cars, Planes and Bicycles. I used Train and Validation images from the COCO2017 dataset (https://cocodataset.org/#download). Each of these images' name is a unique ID and the dataset is accompanied by "annotations". These are .json files which link each image's unique ID with a list of all the objects in that image, as well as the outlines of these objects as sets of (x, y) points (coordinates). One "annotations" file can contain the IDs and properties of thousands of images. 

I used the SAMA-COCO annotations (https://www.sama.com/sama-coco-dataset), as they provide object outlines with higher detail compared to the stock COCO2017 annotations. This dataset's annotations contains several .json files. I started by merging all the "training" immages' annotations into one .json file, and all the "validation" images' annotations into one .json file.

<details>
  <summary>🔹 Click for Code </summary>
  
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

The COCO2017 dataset consists of around 181,000 images. For my training purposes, I only wanted the images that contained Cats, Cars, Planes, and Bicycles. I went through the .json "annotations" file, and extracted the image IDs and object details for only the pictures from the COCO dataset that contained the desired objects. I used that information to create new filtered "annotations" files (containing only the image IDs and details for the desired categories), and duplicated the images of interest into a new folder. That is how I created my custom training dataset's images and annotations.

<details>
  <summary>🔹Click for Code</summary>
  
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

The YOLOv8s-seg model cannot read through the .json "annotations" file that contains all the image IDs and their attributes. It needs a unique .txt text file for each image, that is named exactly the same as the image ID, and contains the details of all objects contained in that specific image. Using a script by z00bean (https://github.com/z00bean/coco2yolo-seg), i went through my filtered dataset "annotations" and created a text file for each of my filtered images, with each of the text files containing all the attributes of its contained objects. This is z00bean's script:

<details>
  <summary>🔹Click for Code</summary>

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
</details>

During the training procedure, the model passes through the large set of train images, examining each picture and its corresponding .txt file. After a full pass, its performance is evaluated by comparing its own inference on objects of the validation images, with the exact attributes of these objects as defines in the .txt annotation files. This cycle constitutes one "epoch". I trained my model on 100 epochs.

<details>
  <summary>🔹Click for Code</summary>

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
</details>

<h1 align="center"></h1>

<h3 align="center"><b>🔺USING THE TRAINED YOLOv8s-seg MODEL🔺</b></h3>

After training my model, I uploaded the weights (model's training knowledge) on my gihub so it can be easily accessible by anyone who wants to download my code and try this for themselves. I set options within my script to use the weights directly from my github link (stores the weights on a temp file and deletes them after the program executes), and to utilize the system's GPU if it's available as it offers better performance compared to the CPU. 

<details>
  <summary>🔹Click for Code</summary>

  ```python

  import cv2
  import numpy as np
  import torch
  import matplotlib.pyplot as plt
  from ultralytics import YOLO
  from matplotlib.ticker import FormatStrFormatter
  from manim import *
  import requests
  import tempfile
  import os
  from io import BytesIO
  
  # URL of the YOLOv8 model
  model_url = "https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/AI%20Model/best.pt"
  
  # Function to download the model file from a URL and return its path
  def download_model_to_tempfile(url):
      response = requests.get(url)
      response.raise_for_status()  # Check for HTTP errors
      
      with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as temp_file:
          temp_file.write(response.content)
          temp_file.flush()  # Ensure all data is written to disk
          temp_file_path = temp_file.name
          print(f"Image Segmentation Model Downloaded Location: {os.path.dirname(temp_file_path)}")  # Print the directory of the temp file
          return temp_file_path
  
  # Set Device to GPU or CPU, according to system specs
  if torch.cuda.is_available() == True:
      device = torch.device("cuda")
      print("")
      print(torch.cuda.get_device_name(device))
      print("")
  else:
      device = torch.device("cpu")
      print("")
      print("Using CPU")
      print("")
  
  # Download the model file to a temporary file
  temp_model_path = download_model_to_tempfile(model_url)
  
  # Load YOLOv8 model from the temporary file
  #model = YOLO(temp_model_path)  # Explicitly define the task
  model = YOLO(temp_model_path)
  model.to(device)
  
  # Clean up: Delete the temporary file
  os.remove(temp_model_path)
  ```
</details>

The user can set the main variable at the top of the script to the image they wish to analyze (options are: 1 for Cat, 2 for Car, 3 for Plane, 4 for Bicycle).

<details>
  <summary>🔹Click for Code</summary>

  ```python

##### DEFINE THIS VARIABLE BEFORE STARTING THE PROGRAM ####
#                                                         #
selected_image_for_processing = 1                         #
#                                                         #
#                       # 1 = CAT                         #
#                       # 2 = CAR                         #
#                       # 3 = PLANE                       #
#                       # 4 = BICYCLE                     #
#                                                         #
###########################################################
  ```
</details>

The script automatically loads the appropriate picture using a link to my github (no need to download the picture). The image is read from the URL in a raw byte data form, where the information regarding the image's colors cannot be interpreted by a human or the AI model. I converted each of these raw bytes to an integer using numpy arrays. That way, the information for the colors of each image was represented by a number (0-255 BGR form, converted to RGB) that could be used to display the image, and also to be processed by the AI model. It is important to note that this decoding process resulted in a flattened 1-D array where the Red, Green and Blue values for each pixel were scattered and not colleceted into individual packets of [Red,Green,Blue]. In order to reformat these numbers in a way that would allow me to display and analyze each image using AI, I used CV2.

I provided CV2 with the above decoded integer RGB values representing the image, and it created an outter array of dimensions equal to the image's resolution (for example for a 640x640 pixel image, the corresponding CV2 array would have 640 columns and 640 rows). Each entry in this outter array represented each of the picture's pixels. Each one of these entries (each pixel), was defined as an individual list of 3 elements. Element 1 was the intensity of Red color for that pixel. Element 2 was the intensity of Green color for that pixel, and element 3 was the intensity of Blue color for that pixel. Thus, each entry of the outter 640 x 640 array represented the color for each pixel in the given image. This is a very important detail, as it means that the image was represented by a tensor. 

After this important converson, I defined a function which set all the graph and animation parameters for the specific picture. The selected and reformated image is presented to the user every time the script is ran.

NOTE: to allow the program to continue, any plot that pops up needs to be closed first. 

<details>
  <summary>🔹Click for Code</summary>

  ```python
  def selection(choice):
      if choice == 1:
          print("")
          print("Cat Selected")
          print("")
          url = "https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Images%20(Input)/cat.jpg"
          zoomval = 800000
          threshold = 600
          anim_dur = 2.95
          l = 700
          h = 700
          l1 = 0.8*1.5*l
          h1 = 0.8*h
      elif choice == 2:
          print("")
          print("Car Selected")
          print("")
          url = "https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Images%20(Input)/car.jpg"
          zoomval = 800000
          threshold = 1500
          anim_dur = 3.52
          l = 700
          h = 700    
          l1 = l
          h1 = h    
      elif choice == 3:
          print("")
          print("Plane Selected")
          print("")
          url = "https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Images%20(Input)/plane.jpg"
          zoomval = 400000
          threshold = 200
          anim_dur = 2.87
          l = 550
          h = 550
          l1 = l
          h1 = h
      elif choice == 4:
          print("")
          print("Bicycle Selected")
          print("")
          url = "https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Images%20(Input)/bicycle.jpg"
          zoomval = 800000
          threshold = 1000
          anim_dur = 5.15
          l = 450
          h = 450
          l1 = 0.85*1.25*l
          h1 = 0.85*h
      return(url, zoomval, threshold, anim_dur, l, h, l1, h1)
  
  selection_result = selection(selected_image_for_processing)
  
  url = selection_result[0]
  zoomval = selection_result[1]
  threshold = selection_result[2]
  anim_dur = selection_result[3]
  l = selection_result[4]
  h = selection_result[5]
  l1 = selection_result[6]
  h1 = selection_result[7] 
  ```
</details>

<table>
  <tr>
    <td><img src="https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Images%20(Input)/cat.jpg" alt="Cat" width="250"/></td>
    <td><img src="https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Images%20(Input)/car.jpg" alt="Car" width="250"/></td>
    <td><img src="https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Images%20(Input)/plane.jpg" alt="Plane" width="250"/></td>
    <td><img src="https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Images%20(Input)/bicycle.jpg" alt="Bicycle" width="250"/></td>
  </tr>
</table>

After the image's conversion to a tensor, I changed its dimensions to 640x640 again using CV2, so that YOLOv8_seg would be able to analyze it. Usually, AI models are made to analyze lists of vectors (tensors). This is why it is very important that I converted my images to tensors before feeding them into the AI model.

I converted my image tensor into a pytorch tensor. While doing so, I defined the RGB values first, then the number of tensor columns, then the number of tensor's rows (it does not change the RGB number values, but it does change the order they appear in the tensor itself. It rearranges the tensor in a non-inutitive way that is required for the model to read the data). I also defined the RGB values as decimals (float) so that the model would get a more accurate understading of my pictures' colors. I defined the reformatted tensor as a batch of size 1 (also a technical requirement for my model to be able to process the data). Lastly, I divided the tensor by 255 in order to have the range of RGB values describing the image set between 0 and 1 (last technical requirement for the model). 

These are very important parameters that the model needs to have defined correctly, in order to be able to analyze the picture.

The model processes the image and returns a "segmentation mask". A "segmentation mask" is a 2-D array of 0s and 1s. The dimensions of this array are exactly the same as the dimensions of the original image. The 0s and 1s represent if the object is present on that part of the image (1) or not (0). The model determines the location of every pixel that belongs to the object, and places a "1" at the location of every such pixel in a 2-D array having the same height and width as the picture. For further processing, these masks need to be transfered to the CPU, if GPU was used for object detection.

<details>
  <summary>🔹Click for Code</summary>

  ```python

# Resize the image to 640x640, to be compatible with YOLOv8
input_image = cv2.resize(input_image, (640, 640))

# Convert the image to tensor to be processed by YOLOv8
input_image = torch.tensor(input_image).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

# Process the image with YOLOv8
results = model(input_image)[0]

# Extract the segmentation masks
masks = results.masks.data.cpu().numpy()  # Move to CPU and convert to numpy

  ```
</details>

Since I resized the image to 640x640 before passing it to the model, the resulting dimensions for the object's location array (segmentation mask) are also 640x640. I resized the mask to the original image dimensions using nearest neighbor interpolation. The new pixel's value (0 or 1) is assigned the value of the closest pixel from the original image. It allowed me to scale the mask to the original, larger dimensions by transfering the information regarding object location in an accurate way. The new pixels that fill in the empty space of the larger image, are made to correspond to the closest pixels to that empty space, as defined in the original image (mask). 

I went through all the values in the properly sized segmentation mask, and set all the 1s to 255, which corresponds to white. The segmentation mask was read by CV2, which identified its entries as black (0) and white (255) pixels. In this way I turned the segmentation mask from a numeric object filled with 0s and 1s, to an image that can be displayed. This image showed the exact area of the detected object in white color, overlayed on a dark background. Using CV2, given the sharp contrast between the area occupied by the object, and its background, I ran contour detection for a reasonably accurate approximation outside contours. This contour detection returns a list of (x, y) points that correspond to the outline of the detected object. I created a new empty image, where I drew the contour using the (x, y) points, and filled it in with red color. I then overlayed this image on the original image, with the filled outline having about 30% transparency. This image is returned to the user.

NOTE: As stated previously, the user needs to close this image for the program to continue.

<details>
  <summary>🔹Click for Code</summary>

  ```python
  # Loop through each resized mask and draw filled translucent contours
  for mask in masks_resized:
      binary_mask = (mask > 0.5).astype(np.uint8) * 255  # Convert to binary mask
  
      # Create an empty image for the mask with the same size as the original image
      filled_mask = np.zeros_like(image, dtype=np.uint8)
  
      # Find contours
      contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  
      # Fill the contours with a translucent color (e.g., red with some transparency)
      for contour in contours:
          cv2.drawContours(filled_mask, [contour], -1, (0, 0, 255), thickness=cv2.FILLED)  # Red fill
      
      # Blend the filled mask with the original image
      translucent_fill = cv2.addWeighted(image, 1, filled_mask, 0.7, 0)  # Adjust alpha and beta for translucency
  ```
</details>

<table>
  <tr>
    <td><img src="https://github.com/NikolaosProjects/AI-Object-Outline-and-Animation/blob/main/All%20Project%20Files/Results/1.%20Cat/Cat%20AI%20Highlighted.png" alt="Cat AI Highlighted" style="width: 250px;"></td>
    <td><img src="https://github.com/NikolaosProjects/AI-Object-Outline-and-Animation/blob/main/All%20Project%20Files/Results/2.%20Car/Car%20AI%20Highlighted.png" alt="Car AI Highlighted" style="width: 250px;"></td>
    <td><img src="https://github.com/NikolaosProjects/AI-Object-Outline-and-Animation/blob/main/All%20Project%20Files/Results/3.%20Plane/Plane%20AI%20Highlighted.png" alt="Plane AI Highlighted" style="width: 250px;"></td>
    <td><img src="https://github.com/NikolaosProjects/AI-Object-Outline-and-Animation/blob/main/All%20Project%20Files/Results/4.%20Bicycle/Bicycle%20AI%20Highlighted.png" alt="Bicycle AI Highlighted" style="width: 250px;"></td>
  </tr>
</table>

The last thing I wanted to do with my outline points, was to convert them to complex (imaginary) numbers. That would allow me to perform Fourier Analysis on the outlines and be able to extract their fourier coefficients and frequencies. After de-nesting the original contour array, I converted each contour point to an imaginary point, and added it to a new array which would be used for fourier analysis

<details>
  <summary>🔹Click for Code</summary>

  ```python
      # Convert contours to complex numbers for Fourier analysis
      for contour in contours:
          contour_points = contour.reshape(-1, 2)  # Flatten the contour
          complex_points = [complex(x, y) for x, y in contour_points]  # Convert to complex numbers
          edges.extend(complex_points)  # Add to the list
  ```
</details>

<h3 align="center"><b>🔺FOURIER ANALYSIS🔺</b></h3>

Since the outline of the object at this stage is nothing but a curve in the 2D complex plane, its shape (like any other curve) can be represented as a superposition of sine and cosine waves. Namely, by their corresponding Fourier Series. This is true for any curve. The arguments and amplitudes of these sine and cosine curves can be determined via Fourier analysis. Numpy offers a module that applies the FFT (Fast Fourier Transform) algorithm to 2D arrays, and extracts the shape's Fourier coefficients and frequencies.

The superposition of sine and cosine graphs is but a large sum of these two functions. In the complex plane, addition of sines and cosines can also be represented by complex exponential functions of magnitudes and frequencies as provided by FFT. Complex exponential functions in turn can be represented by rotating vectors in the complex plane. This means that the outline can be approximated via the tip to tail addition of the vectors, with magnitudes corresponding to the FFT's x and y values (magnitudes), and frequencies corresponding to the FFT's frequencies. 

I applied the FFT algorithm and extracted the outline's fourier coefficients in the form of x, y points, along with their corresponding frequencies. I made sure to center these frequencies and their coefficients in a symmetric manner around f = 0hz.

<details>
  <summary>🔹Click for Code</summary>

  ```python
  from b_MODEL_YoloV8 import *
  
  #placing the extracted image around the origin of the complex plane
  centroid = np.mean(edges)
  edges = edges - centroid
  
  #Get coordinates (x = fft_result.real, y = fft_result.imag) and phase (phase = np.angle(fft_result)) of the fourier coefficients
  fft_result = np.fft.fft(edges)
  
  #Get the corresponding frequencies of the fourier coefficients
  frequencies = np.fft.fftfreq(len(edges))
  
  #EXTRACTING RELEVANT PARAMETERS FROM FFT#
  
  #Define empty lists for coordinates (vectors), magnitude & phase (phase list), and frequency (frequencies) of fourier coefficients
  phase = []
  vectors = []
  frequency = []
  magnitudes = []
  
  #Keeping only the first few coefficients#
  for i in range(len(fft_result)):
      if abs(fft_result[i]) > threshold: #coefficient threshold
          phase.append(np.angle(fft_result[i]))
          vectors.append([fft_result[i].real, fft_result[i].imag]) #used to define the complex exponentials as vectors. It is a convenient representation as the coordinates allow for easy definition of the vectors in manim (using Vector([x,y]), with the phase and magnitudes easily defined in C using only x and y)
          frequency.append(frequencies[i]) #used for determining how fast the complex exponential (vector) should rotate in the subsequent animation
          magnitudes.append(np.abs(fft_result[i])) #used for plotting the distribution of fourier coefficients in frequency-space
  
  #CENTERING THE DATA (Frequency list & Vector List) AT THE 0 FREQUENCY COMPONENT#
  
  data = np.array(np.fft.fftshift(vectors))
  f = np.array(np.fft.fftshift(frequency))
  ```
</details>

The outline of the object in the form of complex points, and the distribution of fourier coefficients' magnitude given their frequency are returned to the user.

<details>
  <summary>🔹Click for Code</summary>

  ```python
  #PLOTTING THE ORIGINAL SHAPE AND THE DISTRIBUTION OF COEFFICIENT MAGNITUDES VS FREQUENCY#
  
  fig, ax = plt.subplots(figsize = (width/l1, height/h1))
  ax.set_facecolor('white')
  ax.scatter(edges.real, -edges.imag, color='black', s=0.2, linestyle = "dotted")
  ax.spines['top'].set_color('black')
  ax.spines['right'].set_color('black')
  ax.spines['left'].set_color('black')
  ax.spines['bottom'].set_color('black')
  ax.spines['left'].set_linestyle('dotted')
  ax.spines['bottom'].set_linestyle('dotted')
  ax.set_xlim(min(edges.real) - 500, max(edges.real) + 500)
  ax.set_ylim(min(edges.imag) - 500, max(edges.imag) + 500)
  ax.set_title('Outline in Complex Plane', color='black', fontsize=10, pad=20, y = 1, x = 0.5)
  plt.tick_params(axis='both', which='major', labelsize=8, colors='black')
  plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f i'))
  plt.show()
  
  #Fourier coefficients
  fig, ax = plt.subplots(figsize = (width/l, height/h))
  ax.set_facecolor('white')
  marker, stemline, baseline = ax.stem(f, magnitudes)
  plt.setp(stemline, linewidth = 2)
  plt.setp(marker, markersize = 0.01)
  plt.setp(baseline, linewidth = 0.001)
  ax.set_xlabel('Frequency', color = "black")
  ax.set_ylabel('Magnitude', color = "black")
  ax.set_title(str(len(f)) + " " + 'Total Fourier Coefficients', color = "Black")
  ax.tick_params(axis='both', colors='black')
  plt.show()
  ```
</details>


<table>
  <tr>
    <td><img src="https://github.com/NikolaosProjects/AI-Object-Outline-and-Animation/blob/main/All%20Project%20Files/Results/1.%20Cat/Cat%20Complex%20Plane.png" alt="Cat Complex Plane" style="width: 250px;"></td>
    <td><img src="https://github.com/NikolaosProjects/AI-Object-Outline-and-Animation/blob/main/All%20Project%20Files/Results/2.%20Car/Car%20Complex%20Plane.png" alt="Car Complex Plane" style="width: 250px;"></td>
    <td><img src="https://github.com/NikolaosProjects/AI-Object-Outline-and-Animation/blob/main/All%20Project%20Files/Results/3.%20Plane/Plane%20Complex%20Plane.png" alt="Plane Complex Plane" style="width: 250px;"></td>
    <td><img src="https://github.com/NikolaosProjects/AI-Object-Outline-and-Animation/blob/main/All%20Project%20Files/Results/4.%20Bicycle/Bicycle%20Complex%20Plane.png" alt="Bicycle Complex Plane" style="width: 250px;"></td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="https://github.com/NikolaosProjects/AI-Object-Outline-and-Animation/blob/main/All%20Project%20Files/Results/1.%20Cat/Cat%20Fourier%20Coefficients.png" alt="Cat Fourier Coefficients" style="width: 250px;"></td>
    <td><img src="https://github.com/NikolaosProjects/AI-Object-Outline-and-Animation/blob/main/All%20Project%20Files/Results/2.%20Car/Car%20Fourier%20Coefficients.png" alt="Car Fourier Coefficients" style="width: 250px;"></td>
    <td><img src="https://github.com/NikolaosProjects/AI-Object-Outline-and-Animation/blob/main/All%20Project%20Files/Results/3.%20Plane/Plane%20Fourier%20Coefficients.png" alt="Plane Fourier Coefficients" style="width: 250px;"></td>
    <td><img src="https://github.com/NikolaosProjects/AI-Object-Outline-and-Animation/blob/main/All%20Project%20Files/Results/4.%20Bicycle/Bicycle%20Fourier%20Coefficients.png" alt="Bicycle Fourier Coefficients" style="width: 250px;"></td>
  </tr>
</table>



<h3 align="center"><b>🔺OUTLINE ANIMATION🔺</b></h3>

As explained earlier, since the Fourier coefficients are vectors in the complex plane, and have a specific frequency associated with them, I can make them rotate. if I pair them tip to tail, and make each vector rotate about the point of the previous vector, tracing the path of the last vector's tip results in tracing the object's outline, given a sufficiencly large number Fourier coefficients (number of rotating vectors), to ensure accuracy.

Using the python module Manim, it is relatively easy to achieve this. I used the Fourier coefficients' coordinates to define a vector corresponding to each coefficient, and then define its angular velocity (rate of rotation) using $$ omega = 2*pi*f $$. Manim creates an animation by updating a $$dt$$ variable, where $$dt$$ is the amount of time in seconds between frames (60fps means dt = 1/60s). To make the vectors rotate, I needed to update their position after time $$dt$$. To do that, I made them rotate by an angle theta, defined by their frequency, after every time $$dt$$. Mathematically, a vector roating at frequency $$f$$, after time duration $$dt$$, will roate by an angle: $$theta = omega*dt => theta = 2*pi*f*dt$$.

After each frame, I updated the position of each vector, by rotating it with its corresponding angle $$theta$$. To make the animations go faster, I multiplied the frequency of each vector by 500 (it makes every signle vector rotate 500 times faster).

<details>
  <summary>🔹Click for Code</summary>

  ```python



  ```
</details>

<table>
  <tr>
    <td><img src="https://github.com/NikolaosProjects/AI-Object-Outline-and-Animation/blob/main/All%20Project%20Files/Results/1.%20Cat/Cat%20Outline%20Animation.gif" alt="Cat Outline Animation" style="width: 200px;"></td>
    <td><img src="https://github.com/NikolaosProjects/AI-Object-Outline-and-Animation/blob/main/All%20Project%20Files/Results/2.%20Car/Car%20Outline%20Animation.gif" alt="Car Outline Animation" style="width: 200px;"></td>
    <td><img src="https://github.com/NikolaosProjects/AI-Object-Outline-and-Animation/blob/main/All%20Project%20Files/Results/3.%20Plane/Plane%20Outline%20Animation.gif" alt="Plane Outline Animation" style="width: 200px;"></td>
    <td><img src="https://github.com/NikolaosProjects/AI-Object-Outline-and-Animation/blob/main/All%20Project%20Files/Results/4.%20Bicycle/Bicycle%20Outline%20Animation.gif" alt="Bicycle Outline Animation" style="width: 200px;"></td>
  </tr>
</table>

