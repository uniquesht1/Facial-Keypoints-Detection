import numpy as np
import torch
from torchvision import transforms
from facial_key_point.model.vgg import get_model  # Import your VGG model architecture

class FacialKeyPointDetection:
    def __init__(self) -> None:
        # Set device to GPU if available, otherwise fallback to CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the model architecture
        self.model = get_model(device=self.device)  # Use your VGG model

        # Load the state dictionary (weights)
        model_path = r'D:\online class\DeepLearning\Facial key point\dump\version_1\model.pth'
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

    def predict(self, image):
        # Preprocess the image and get a displayable version
        img, img_disp = self.preprocess(image)

        # Ensure the image tensor is on the same device as the model
        img = img.to(self.device)

        # Predict keypoints using the model
        with torch.no_grad():
            img = img.unsqueeze(0)  # Add batch dimension (1, 3, 224, 224)
            kps = self.model(img).flatten().detach().cpu()  # Flatten the keypoints

        # Post-process keypoints to match the image dimensions
        kp_x, kp_y = self.postprocess(img_disp, kps)
        return img_disp, (kp_x, kp_y)
    
    def preprocess(self, img):
        # Resize and normalize the image
        img = img.resize((224, 224))
        img_disp = np.asarray(img) / 255.0  # Display image is not normalized
        
        # Prepare the tensor for model input
        img_tensor = torch.tensor(img_disp).permute(2, 0, 1).float()  # Convert to tensor and permute
        img_tensor = self.normalize(img_tensor).float()  # Normalize the image
        
        # Return the tensor for model input and the displayable image
        return img_tensor, img_disp

    def postprocess(self, img, kps):
        # Convert keypoints to image coordinates
        img = np.array(img)
        width, height, _ = img.shape

        kp_x = kps[:68] * width
        kp_y = kps[68:] * height
        return kp_x, kp_y
