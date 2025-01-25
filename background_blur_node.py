import cv2
import numpy as np
import torch

class BackgroundBlurNode:
    def __init__(self):
        # default values
        self.blur_radius = 15
        self.threshold = 128
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_radius": ("INT", {
                    "default": 15,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "threshold": ("INT", {
                    "default": 128,
                    "min": 0,
                    "max": 255,
                    "step": 1
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_background_blur"
    CATEGORY = "image/processing"

    def apply_background_blur(self, image, blur_radius, threshold):
        # Convert from torch tensor to numpy array
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if len(image.shape) == 4:
            image = image[0]
        
        # Convert from CHW to HWC format
        image = np.transpose(image, (1, 2, 0))
        
        # Convert to uint8 for OpenCV processing
        image = (image * 255).astype(np.uint8)
        
        # Create a simple mask based on pixel intensity
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Apply Gaussian blur to the background
        blurred = cv2.GaussianBlur(image, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
        
        # Combine original foreground with blurred background
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        result = np.where(mask == 255, image, blurred)
        
        # Convert back to CHW format and normalize to 0-1 range
        result = np.transpose(result, (2, 0, 1)).astype(np.float32) / 255.0
        
        # Convert to torch tensor
        result = torch.from_numpy(result).unsqueeze(0)
        
        return (result,)

# Node registration
NODE_CLASS_MAPPINGS = {
    "BackgroundBlur": BackgroundBlurNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BackgroundBlur": "Background Blur"
}