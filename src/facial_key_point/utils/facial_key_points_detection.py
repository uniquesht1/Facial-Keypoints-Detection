
#For single face detection

import numpy as np
import torch
from torchvision import transforms
from facial_key_point.model.vgg import get_model  # Import your VGG model architecture


# class FacialKeyPointDetection:
#     def __init__(self) -> None:
#         # Set device to GPU if available, otherwise fallback to CPU
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # Initialize the model architecture
#         self.model = get_model(device=self.device)  # Use your VGG model

#         # Load the state dictionary (weights)
#         model_path = r'D:\online class\DeepLearning\Facial key point\dump\version_1\model.pth'
#         state_dict = torch.load(model_path, map_location=self.device)
#         self.model.load_state_dict(state_dict)

#         # Set model to evaluation mode
#         self.model.to(self.device)  # Move model to GPU (or CPU if unavailable)
#         self.model.eval()

#         # Normalization for image preprocessing
#         self.normalize = transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )

#     def predict(self, image):
#         # Preprocess the image and get a displayable version
#         img, img_disp = self.preprocess(image)

#         # Ensure the image tensor is on the same device as the model
#         img = img.to(self.device)

#         # Predict keypoints using the model
#         with torch.no_grad():
#             img = img.unsqueeze(0)  # Add batch dimension (1, 3, 224, 224)
#             kps = self.model(img).flatten().detach().cpu()  # Flatten the keypoints

#         # Post-process keypoints to match the image dimensions
#         kp_x, kp_y = self.postprocess(img_disp, kps)
#         return img_disp, (kp_x, kp_y)
    
#     def preprocess(self, img):
#         # Resize and normalize the image
#         img = img.resize((224, 224))
#         img_disp = np.asarray(img) / 255.0  # Display image is not normalized
        
#         # Prepare the tensor for model input
#         img_tensor = torch.tensor(img_disp).permute(2, 0, 1).float()  # Convert to tensor and permute
#         img_tensor = self.normalize(img_tensor).float()  # Normalize the image
        
#         # Return the tensor for model input and the displayable image
#         return img_tensor, img_disp

#     def postprocess(self, img, kps):
#         # Convert keypoints to image coordinates
#         img = np.array(img)
#         width, height, _ = img.shape

#         kp_x = kps[:68] * width
#         kp_y = kps[68:] * height
#         return kp_x, kp_y

# for multiple face detection

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import cv2
from facial_key_point.model.vgg import get_model  # Import your VGG model architecture


class FacialKeyPointDetection:
    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> None:
        # Set device to GPU if available, otherwise fallback to CPU
        self.device = device

        # Initialize the model architecture
        self.model = get_model(device=self.device)  # Use your VGG model

        # Load the state dictionary (weights)
        model_path = r'D://online class//DeepLearning//Facial key point//dump//version_1//model.pth'
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        # Set model to evaluation mode
        self.model.to(self.device)  # Move model to GPU (or CPU if unavailable)
        self.model.eval()

        # Normalization for image preprocessing
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.model_input_size = (224, 224)  # Assuming 224x224 as the model input size

    def process_image(self, img_path):
        # Load the image and process faces
        img, results = self.get_img(img_path)
        return img, results

    def get_img(self, img_path):
        # Load and convert image to RGB
        img = Image.open(img_path).convert('RGB')
        original_img = np.array(img)  
        
        # Resize and normalize the image for model input
        img = img.resize(self.model_input_size, Image.Resampling.BILINEAR)
        img = np.asarray(img) / 255.0
        img_tensor = torch.tensor(img).permute(2, 0, 1)  #
        img_tensor = self.normalize(img_tensor).float()

        # Detect faces in the image
        faces = self.detect_face(original_img)

        results = []
        for (x, y, w, h) in faces:
            face_img = Image.fromarray(original_img[y:y + h, x:x + w])
            face_tensor = self.preprocess_face(face_img).to(self.device)
            kp = self.get_keypoints_for_face(face_tensor)

            kp_x = np.array(kp[:68]) * w + x
            kp_y = np.array(kp[68:]) * h + y

            results.append((x, y, w, h, kp_x, kp_y, face_img))
        return original_img, results

    def preprocess_face(self, face_img):
        # Resize and normalize face for model input
        face_img = face_img.resize(self.model_input_size, Image.Resampling.BILINEAR)
        face_img = np.asarray(face_img) / 255.0
        face_tensor = torch.tensor(face_img).permute(2, 0, 1)
        face_tensor = self.normalize(face_tensor).float()
        return face_tensor

    def get_keypoints_for_face(self, face_tensor):
        # Predict keypoints for the detected face
        with torch.no_grad():
            face_tensor = face_tensor.unsqueeze(0)  
            keypoints = self.model(face_tensor)  

        keypoints = keypoints.squeeze().cpu().numpy()  
        return keypoints

    def detect_face(self, img_array):
        # Convert image to uint8 if necessary
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)

        # Convert image to grayscale and detect faces
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def visualize_keypoints_on_original(self, original_image, results, save_path=None):
        # Visualize the keypoints on the original image
        plt.figure(figsize=(10, 10))
        plt.imshow(original_image)

        for (x, y, w, h, kp_x, kp_y, _) in results:
            plt.scatter(kp_x, kp_y, c='yellow', s=10, marker='o') 

        plt.axis('off')  
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')  
            print(f"Results saved to {save_path}")
        
        plt.show()  


# Usage example:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_path = 'multi-face-keypoint-detection.jpg'

# Initialize the detector
detector = FacialKeyPointDetection(device=device)

# Process image and get keypoints
image, results = detector.process_image(img_path)

# Save the visualization
save_path = './output_image_with_keypoints.jpg'
detector.visualize_keypoints_on_original(image, results, save_path=save_path)
