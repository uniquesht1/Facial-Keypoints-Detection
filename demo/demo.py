import streamlit as st
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

# Append the correct path to sys.path to import from 'src'
sys.path.append(r'D://online class//DeepLearning//Facial key point//src')

# Import the FacialKeyPointDetection class
from facial_key_point.utils.facial_key_points_detection import FacialKeyPointDetection

# Streamlit app for Facial Keypoint Detection
st.markdown('## Facial Key Point Detection')

# Upload an image
image = st.file_uploader('Upload a Facial Image', ['jpg', 'png', 'jpeg'], accept_multiple_files=False)
if image is not None:
    image = Image.open(image).convert('RGB')
    st.image(image)

    # Initialize the detector
    detector = FacialKeyPointDetection()

    # Make predictions
    image_disp, kp = detector.predict(image)

    # Display the image and predicted keypoints
    fig = plt.figure()
    plt.imshow(np.array(image_disp))  # Display the original image
    plt.scatter(kp[0], kp[1], s=4, c='r')  # Plot the keypoints
    st.pyplot(fig)
