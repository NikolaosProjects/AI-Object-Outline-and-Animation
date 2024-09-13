<h1 align="center"><b>Animating Outlines: AI & Fourier Series</b></h1>

<h3 align="center"><b>CAT</b></h3>
<div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
  <img src="https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Results/1.%20Cat/Cat%20Image.png" alt="Cat Image" width="49%" height="300px" style="object-fit: cover;">
  <img src="https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Results/1.%20Cat/Cat%20Outline%20Animation.gif" alt="Cat Animation Gif" width="49%" height="300px" style="object-fit: cover;">
</div>

<h1 align="center"></h1>

<h3 align="center"><b>CAR</b></h3>
<div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
  <img src="https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Results/2.%20Car/Car%20Image.png" alt="Car Image" width="49%" height="300px" style="object-fit: cover;">
  <img src="https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Results/2.%20Car/Car%20Outline%20Animation.gif" alt="Car Animation Gif" width="49%" height="300px" style="object-fit: cover;">
</div>

<h1 align="center"></h1>

<h3 align="center"><b>PLANE</b></h3>
<div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
  <img src="https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Results/3.%20Plane/Plane%20Image.png" alt="Plane Image" width="49%" height="300px" style="object-fit: cover;">
  <img src="https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Results/3.%20Plane/Plane%20Outline%20Animation.gif" alt="Plane Animation Gif" width="49%" height="300px" style="object-fit: cover;">
</div>

<h1 align="center"></h1>

<h3 align="center"><b>BICYCLE</b></h3>
<div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
  <img src="https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Results/4.%20Bicycle/Bicycle%20Image.png" alt="Bicycle Image" width="49%" height="300px" style="object-fit: cover;">
  <img src="https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/All%20Project%20Files/Results/4.%20Bicycle/Bicycle%20Outline%20Animation.gif" alt="Bicycle Animation Gif" width="49%" height="300px" style="object-fit: cover;">
</div>

<h1 align="center"></h1>

<h1 align="center"><b>Project Description</b></h1>

For this project, I trained an image segmentation model to accurately identify and trace the outlines of Cats, Cars, Planes, and Bicycles. I converted these borders into points in the complex plane, and analyzed them using Fourier Analysis. I used the fourier coefficients and their frequencies as rotating vectors, and used their rotation to trace the outline they originated from.

<h1 align="center"></h1>

<h3 align="left"><b>Artificial Intelligence Model</b></h3>

I used the untrained YOLOv8s-seg model from Ultralytics (https://docs.ultralytics.com/tasks/segment), which is an Artificial Intelligence image detection and segmentation model. When provided with an image, it can identify objects from up to 80 different categories. Additionally, it can detect the location of these objects in the image, and identify their exact borders.

<h1 align="center"></h1>

<h3 align="left"><b>Model Training</b></h3>

For my project I wanted to provide my own training to the model. I used the COCO2017 Train and Validation images (https://cocodataset.org/#download). Each of these images has a unique name, and the dataset is accompanied "annotations". These are .json text files that link each image's unique ID with a list of all the objects in that image, as well as the outlines of these objects as sets of (x, y) points (coordinates).

I used the SAMA-COCO annotations (https://www.sama.com/sama-coco-dataset), as they provide object outlines with higher detail compared to the stock COCO2017 annotations.

The COCO2017 dataset consists of around 181,000 images. For my training purposes, I only wanted the images that contained Cats, Cars, Planes, and Bicycles. I this went through the annotations file, and extracted the image IDs and object details, for only the pictures from the COCO dataset that contained my desired categories. I used that information to create new filtered annotations files (containing only the image IDs and details for my desired categories), and duplicated the images of interest into a new folder. That is how I created my custom dataset. 

The YOLOv8s-seg model cannot read through the .json file that contains all the image IDs and their attributes. It needs a unique text file for each image, that is named exactly the same as the image ID, and contains the details for that specific image. Using a script by z00bean (https://github.com/z00bean/coco2yolo-seg), i went through my filtered dataset annotations and created a text file for each of my filtered images, with each of the text files containing all the object present in the image, and the coordinates of each object's boundary.

During the training procedure, the model passes through the large set of train images, examining each picture and its corresponding .txt file. After a full pass, its performance is evaluated by comparing its own inference on objects of the validation images, with the exact attributes of these objects as defines in the .txt annotation files. This cycle constitutes one "epoch". I trained my model on 100 epochs.


