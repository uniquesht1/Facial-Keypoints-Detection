# Facial Keypoint Detection

This project focuses on detecting facial keypoints from facial images using a deep learning approach with the **VGG16** model. The keypoints detected include important facial landmarks like eyes, nose, and mouth.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model](#model)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Visualizing Results](#visualizing-results)
- [Future Work](#future-work)
- [License](#license)

## Project Overview

Facial keypoints are used in a variety of tasks such as facial recognition, emotion detection, and augmented reality filters. This project implements a **Facial Keypoint Detection** system using the **VGG16** architecture and PyTorch. The model detects 68 keypoints on a face from an input image, which can then be visualized or used in other facial analysis tasks.

## Dataset

The dataset used for this project consists of facial images and their corresponding keypoints (68 facial landmarks). The keypoints are stored as `(x, y)` coordinates for each facial image.

- **Training Dataset**: Located in `data/training_frames_keypoints.csv`
- **Test Dataset**: Located in `data/test_frames_keypoints.csv`

Each dataset contains:
- A path to the image file.
- 136 columns representing the `x` and `y` coordinates for 68 facial keypoints.

## Model

The model is based on **VGG16** pretrained on ImageNet, but modified to output 136 keypoint coordinates (68 keypoints, with 2 values each for x and y).

Key points of the model architecture:
- **Pre-trained VGG16** for transfer learning.
- Modified to output 136 keypoint values.
- Optimized with **Adam optimizer** and **L1 loss** for regression tasks.

## Installation

### Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Numpy
- Pandas
- Matplotlib
- tqdm

### Setup

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/facial-keypoint-detection.git
    cd facial-keypoint-detection
    ```

2. **Install the required dependencies**:

    Create and activate a virtual environment:
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Linux/Mac
    .\env\Scripts\activate  # On Windows
    ```

    Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Training the Model

### DataLoader

The `FaceKeyPointData` class handles image preprocessing and data loading. Images are resized to `224x224`, normalized, and transformed into tensors for model input.

### Training Loop

1. **Set Hyperparameters**:
    ```python
    batch_size = 16
    model_input_size = 224
    n_epoch = 10
    learning_rate = 0.0001
    ```

2. **Train the model**:
   The training loop utilizes `train_batch` and `validation_batch` functions to handle forward and backward passes. The model uses **L1 loss** for regression and updates weights using **Adam optimizer**.
   
   Run the training process:
    ```python
    for epoch in range(1, n_epoch+1):
        # Training and Validation steps
    ```

3. **Track Loss**:
    Loss for both training and validation is tracked over each epoch. Use matplotlib to plot the loss curves.

    ```python
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, test_loss, 'r', label='Test Loss')
    ```

## Visualizing Results

You can visualize the keypoints on the test images using the trained model:

```python
img_index = 30
img = test_data.load_img(img_index)

# Plot original image and the image with predicted keypoints
plt.subplot(121)
plt.title('Original Image')
plt.imshow(img)

plt.subplot(122)
plt.title("Image with Facial Keypoints")
plt.imshow(img)

# Predict and plot keypoints
img_tensor, _ = test_data[img_index]
pred_keypoints = model(img_tensor[None]).flatten().detach().cpu()
plt.scatter(pred_keypoints[:68] * model_input_size, pred_keypoints[68:] * model_input_size, c='y', s=2)
