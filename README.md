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

```
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

The COCO2017 dataset consists of around 181,000 images. For my training purposes, I only wanted the images that contained Cats, Cars, Planes, and Bicycles. I went through the .json "annotations" file, and extracted the image IDs and object details for only the pictures from the COCO dataset that contained the desired objects. I used that information to create new filtered "annotations" files (containing only the image IDs and details for the desired categories), and duplicated the images of interest into a new folder. That is how I created my custom training dataset's images and annotations.

The YOLOv8s-seg model cannot read through the .json "annotations" file that contains all the image IDs and their attributes. It needs a unique .txt text file for each image, that is named exactly the same as the image ID, and contains the details of all objects contained in that specific image. Using a script by z00bean (https://github.com/z00bean/coco2yolo-seg), i went through my filtered dataset "annotations" and created a text file for each of my filtered images, with each of the text files containing all the attributes of its contained objects.

During the training procedure, the model passes through the large set of train images, examining each picture and its corresponding .txt file. After a full pass, its performance is evaluated by comparing its own inference on objects of the validation images, with the exact attributes of these objects as defines in the .txt annotation files. This cycle constitutes one "epoch". I trained my model on 100 epochs.

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
