import os
import folder_paths

import cv2
import numpy as np
import torch

from huggingface_hub import hf_hub_download

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
        
        result = result.astype(np.float32) / 255.0
        
        # Convert to torch tensor
        result = torch.from_numpy(result).unsqueeze(0)
        
        return (result,)

AVAILABLE_MODELS = {
    "RMBG-2.0": {
        "type": "rmbg",
        "repo_id": "briaai/RMBG-2.0",
        "files": {
            "config.json": "config.json",
            "model.safetensors": "model.safetensors",
            "birefnet.py": "birefnet.py",
            "BiRefNet_config.py": "BiRefNet_config.py"
        },
        # models will be saved in models/RMBG/RMBG-2.0
        "cache_dir": "RMBG-2.0"
    }
}    
class BaseRMBGModelLoader:
    def __init__(self):
        self.model = None
        self.base_cache_dir = os.path.join(folder_paths.models_dir, "RMBG")
    
    def get_cache_dir(self, model_name):
        return os.path.join(self.base_cache_dir, 
                            AVAILABLE_MODELS[model_name]['cache_dir'])

    # check if model is cached
    def check_model_cahce(self, model_name):
        model_info = AVAILABLE_MODELS[model_name]
        cache_dir = self.get_cache_dir(model_name)
        # check if cache_dir exists
        if not os.path.exists(cache_dir):
            return False
        
        missing_file = {}
        for file_name in model_info['files'].key():
            if not os.path.exists(os.path.join(cache_dir, model_info['files'][file_name])):
                missing_file[file_name] = model_info['files'][file_name]
        # check if missing_file is empty
        if len(missing_file):
            return False
        
        return True
    
    def download_model(self, model_name):
        model_info = AVAILABLE_MODELS[model_name]
        cache_dir = self.get_cache_dir(model_name)
        
        try:
            os.makedirs(cache_dir, exist_ok=True)
            
            for filename in model_info["files"].keys():
                # Download the file by huggingface
                hf_hub_download(
                    repo_id=model_info["repo_id"],
                    filename=filename,
                    local_dir=cache_dir,
                    local_dir_use_symlinks=False
                )
                    
            return True
            
        except Exception as e:
            print(e)
            return False
        
    def clear_model(self):
        try:
            if self.model is not None:
                # del the model's parameters gradient
                if hasattr(self.model, 'parameters'):
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.zero_()
                            del param.grad
                
                self.model.cpu()
                # del the model
                del self.model
                self.model = None

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                import gc
                gc.collect()

                return True
            
        except Exception as e:
            print(f"Error clearing model: {str(e)}")
            return False
    
# Node registration
NODE_CLASS_MAPPINGS = {
    "BackgroundBlur": BackgroundBlurNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BackgroundBlur": "Background Blur"
}
