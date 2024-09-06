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





##### DEFINE THIS VARIABLE BEFORE STARTING THE PROGRAM ####
#                                                         #
selected_image_for_processing = 4                         #
#                                                         #
#                       # 1 = CAT                         #
#                       # 2 = CAR                         #
#                       # 3 = PLANE                       #
#                       # 4 = BICYCLE                     #
#                                                         #
###########################################################





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

# Set Device to GPU
device = torch.device("cuda")

# URL of the YOLOv8 model
model_url = "https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/AI%20Model/yolov8s-seg.pt"

# Download the model file to a temporary file
temp_model_path = download_model_to_tempfile(model_url)

# Load YOLOv8 model from the temporary file
model = YOLO(temp_model_path, task='detect')  # Explicitly define the task
model.to(device)

# Clean up: Delete the temporary file
os.remove(temp_model_path)

def selection(choice):
    if choice == 1:
        print("")
        print("Cat Selected")
        print("")
        url = "https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/Images/cat.jpg"
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
        url = "https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/Images/car.jpg"
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
        url = "https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/Images/plane.jpg"
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
        url = "https://raw.githubusercontent.com/NikolaosProjects/AI-Object-Outline-and-Animation/main/Images/bicycle.jpg"
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


#load image from github
response = requests.get(url)
image_data = response.content
# Convert the image data to a NumPy array
image_array = np.asarray(bytearray(image_data), dtype=np.uint8)
# Decode the image array using OpenCV
image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
# Correct the image color (when the image is read, its colors are flipped)
input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# show the selected image
height, width, _ = input_image.shape
fig, ax = plt.subplots(figsize = (width/l, height/h))
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.imshow(input_image)
ax.axis('off')
ax.axis('off')
plt.show()

# Resize the image to 640x640, to be compatible with YOLOv8
input_image = cv2.resize(input_image, (640, 640))
# Convert the image to tensor to be processed by YOLOv8
input_image = torch.tensor(input_image).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

# Process the image with YOLOv8
results = model(input_image)[0]

# Extract the segmentation masks
masks = results.masks.data.cpu().numpy()  # Move to CPU and convert to numpy

# Resize the masks back to the original image size
masks_resized = [cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) for mask in masks]

# Define empty list for storing the outline as a set of complex points
edges = []

# Loop through each resized mask and draw filled translucent contours
for mask in masks_resized:
    binary_mask = (mask > 0.5).astype(np.uint8) * 255  # Convert to binary mask

    # Create an empty image for the mask with the same size as the original image
    filled_mask = np.zeros_like(image, dtype=np.uint8)

    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Fill the contours with a translucent color (e.g., red with some transparency)
    for contour in contours:
        cv2.drawContours(filled_mask, [contour], -1, (0, 0, 255), thickness=cv2.FILLED)  # Red fill
    
    # Blend the filled mask with the original image
    translucent_fill = cv2.addWeighted(image, 1, filled_mask, 0.7, 0)  # Adjust alpha and beta for translucency

    # Convert contours to complex numbers for Fourier analysis
    for contour in contours:
        contour_points = contour.reshape(-1, 2)  # Flatten the contour
        complex_points = [complex(x, y) for x, y in contour_points]  # Convert to complex numbers
        edges.extend(complex_points)  # Add to the list

# Plot the result with filled translucent contours
height, width, _ = translucent_fill.shape
fig, ax = plt.subplots(figsize = (width/l, height/h))
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.imshow(cv2.cvtColor(translucent_fill, cv2.COLOR_BGR2RGB))#, aspect='auto')
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.axis('off')
plt.show()

# Convert list to numpy array and save to file
edges = np.array(edges)
