from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import joblib
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.morphology import disk, binary_opening, binary_closing, remove_small_objects
from skimage.measure import label, regionprops
from skimage.color import rgb2gray
from skimage.util import invert
from ultralytics import YOLO

# Load the saved SVM model and label encoder
svm_model = joblib.load('svm_model.pkl')
label_encoder = joblib.load('svm_label_encoder.pkl')

# Load the saved CNN model
cnn_model = YOLO("./cnn_model.pt")

# Placeholder for YOLO model loading (replace with your YOLO loading code if available)
def load_yolo_model():
    # Dummy function to demonstrate structure
    print("YOLO model loaded.")
    return None

yolo_model = load_yolo_model()

# Global variable for the filename
filename = ""

def predict_image(image_path):
    if not image_path:
        svmPredictionLabel.config(text="No file selected! Please choose an image.", fg="red")
        yoloPredictionLabel.config(text="No file selected! Please choose an image.", fg="red")
        return

    # SVM Prediction
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, gray.mean() - gray.std(), gray.mean() + gray.std())  # Apply Canny edge detection

        binarized_edges = edges > 0  # Binarize the edge-detected image
        binarized_edges = remove_small_objects(binarized_edges, min_size=3)
    
        closed_edge = binary_closing(binarized_edges, disk(17))
        filled_segment = cv2.morphologyEx(closed_edge.astype(np.uint8), cv2.MORPH_CLOSE, None)
        
        # Apply the mask to the image
        if len(img.shape) == 3:
            obj_img = img * filled_segment[:, :, np.newaxis]
        else:
            obj_img = img * filled_segment
        
        # Histogram of Oriented Gradients (HOG)
        winSize = (64, 64)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        hog_features = [hog.compute(obj_img).flatten()]

        # Predict the class and confidence
        predicted_class_index = svm_model.predict(hog_features)[0]
        confidence = np.max(svm_model.predict_proba(hog_features))
        predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]

        svmPredictionLabel.config(text=f"SVM Prediction: {predicted_label} ({confidence:.2f})", fg="green")
    except Exception as e:
        print(e)
        svmPredictionLabel.config(text=f"SVM Error: {e}", fg="red")

    # YOLO Prediction
    try:
        labels = ['Bus', 'Car', 'Motorcycle', 'Truck']
        result = cnn_model(image_path)
        predicted_label = labels[result[0].probs.top1]
        confidence = result[0].probs.top1conf.item()

        yoloPredictionLabel.config(text=f"YOLO Prediction: {predicted_label} ({confidence:.2f})", fg="blue")
    except Exception as e:
        yoloPredictionLabel.config(text=f"YOLO Error: {e}", fg="red")

def browseFiles():
    global filename  # Declare filename as a global variable
    filename = filedialog.askopenfilename(
        initialdir="D:/Kuliah/SMT 7/PCD/Tucil/Tucil 4/IF4073-Vehicle-Recognition/dataset/test",
        title="Select a File",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
    )
    
    if filename:
        try:
            # Open the image using Pillow (PIL)
            img = Image.open(filename)
            # img = img.resize((600, 600))  # Resize image to fit the label
            # Convert the image to a Tkinter-compatible format
            img_tk = ImageTk.PhotoImage(img)

            # Update the image label to display the selected image
            imageLabel.config(image=img_tk)
            imageLabel.image = img_tk  # Keep a reference to avoid garbage collection
        except Exception as e:
            print(f"Error loading image: {e}")
    else:
        svmPredictionLabel.config(text="No file chosen. Please select an image.", fg="red")
        yoloPredictionLabel.config(text="No file chosen. Please select an image.", fg="red")

root = Tk()
root.title("Vehicle Recognition")
root.geometry("800x700")
root.configure(bg="#f0f0f0")  # Light grey background for a cleaner look

# Create a header label
headerLabel = Label(root, text="Vehicle Recognition System", font=("Arial", 24, "bold"), bg="#f0f0f0", fg="#333")
headerLabel.pack(pady=20)

# Image display section
imageLabel = Label(root, text="No Image Selected", bg="#ccc", fg="#666")
imageLabel.pack(pady=20)

# Prediction labels
svmPredictionLabel = Label(root, text="SVM Prediction: None", font=("Arial", 14), bg="#f0f0f0", fg="#333")
svmPredictionLabel.pack(pady=10)

yoloPredictionLabel = Label(root, text="YOLO Prediction: None", font=("Arial", 14), bg="#f0f0f0", fg="#333")
yoloPredictionLabel.pack(pady=10)

# Control buttons
controlFrame = Frame(root, bg="#f0f0f0")
controlFrame.pack(pady=20)

inputButton = Button(controlFrame, text="Select Image", width=20, command=browseFiles, bg="#4caf50", fg="white")
inputButton.grid(row=0, column=0, padx=10)

processButton = Button(controlFrame, text="Scan Image", width=20, command=lambda: predict_image(filename), bg="#2196f3", fg="white")
processButton.grid(row=0, column=1, padx=10)

root.mainloop()
