import os
import folder_paths

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageSegmentation

from ..utils.transformation import tensor2pil, pil2tensor

device = "cuda" if torch.cuda.is_available() else "cpu"
class BackgroundBlurNode:
    def __init__(self):
        # default values
        self.blur_radius = 30
        self.threshold = 255
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (list(AVAILABLE_MODELS.keys()), {"default": "RMBG-2.0"}),
                "blur_radius": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "threshold": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "step": 1
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "apply_background_blur"
    CATEGORY = "image/processing"

    def apply_background_blur(self, image, model, blur_radius, threshold):
        
        # Remove background and obtain the foreground and mask
        rmbg_model = RMBGModel()
        foreground_all_channel, fg_mask_0 = rmbg_model.process_image(image, model)
        # fg_mask_0 is the mask of the original foreground image as output
        fg_mask = fg_mask_0.copy()
        # RGBA2RGB
        foreground = foreground_all_channel[:, :, :, :3]

        # Convert from torch tensor to numpy array
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if len(image.shape) == 4:
            image = image[0]

        # Convert to uint8 for OpenCV processing
        image = (image * 255).astype(np.uint8)
        
        # Create a simple mask based on pixel intensity
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, bg_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Apply Gaussian blur to the background
        blurred = cv2.GaussianBlur(image, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0)
        
        # Combine original foreground with blurred background
        bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2RGB)
        background = np.where(bg_mask == 255, image, blurred)
        
        background = background.astype(np.float32) / 255.0
        
        # Convert to torch tensor
        background = torch.from_numpy(background).unsqueeze(0)

        fg_mask = fg_mask.unsqueeze(3)
        fg_mask = fg_mask.repeat(1, 1, 1, 3)

        # Combine foreground and background by dot product
        result = foreground * fg_mask + background * (1 - fg_mask) 
        
        return (result, fg_mask_0)

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
    def check_model_cache(self, model_name):
        model_info = AVAILABLE_MODELS[model_name]
        cache_dir = self.get_cache_dir(model_name)
        # check if cache_dir exists
        if not os.path.exists(cache_dir):
            return False
        
        missing_file = {}
        for file_name in model_info['files'].keys():
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
            
        except Exception as e:
            print(f"Error clearing model: {str(e)}")

class RMBGModel(BaseRMBGModelLoader):
    def __init__(self):
        super().__init__()

    def load_model(self, model_name):
        self.clear_model()
        
        cache_dir = self.get_cache_dir(model_name)
        self.model = AutoModelForImageSegmentation.from_pretrained(cache_dir,trust_remote_code=True,local_files_only=True)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False
        
        torch.set_float32_matmul_precision('high')
        self.model.to(device)

    def preprocess_image(self, image, model_name):
        images = [image] if not isinstance (image, list) else image
        if not self.check_model_cache(model_name):
            print(f"Model {model_name} not found in cache. Downloading...")
            if not self.download_model(model_name):
                raise RuntimeError(f"Failed to download model {model_name}")
        print(f"Preprocess image and model {model_name} downloaded successfully")
        return images

    def remove_background(self, background, model_name):
        try:
           self.load_model(model_name)

           transform_image = transforms.Compose(
                [
                    transforms.Resize((1024, 1024)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
           
           imgs = [background] if isinstance(background, torch.Tensor) and len(background.shape) == 3 else background

           original_sizes = [tensor2pil(img).size for img in imgs]
           
           # batch processing imgs to accelerate removing background   
           input_tensors = [transform_image(tensor2pil(img)).unsqueeze(0) for img in imgs]
           input_batch = torch.cat(input_tensors, dim=0).to(device)

           with torch.no_grad():
                preds = self.model(input_batch)[-1].sigmoid().cpu()
                masks = []

                for _, (pred, (orig_w, orig_h)) in enumerate(zip(preds, original_sizes)):
                    pred = pred.squeeze()
                    pred = torch.clamp(pred, 0, 1)    
                    # Resize back to original dimensions
                    pred = F.interpolate(pred.unsqueeze(0).unsqueeze(0),
                                            size=(orig_h, orig_w),
                                            mode='bilinear').squeeze()
                    
                    masks.append(tensor2pil(pred))

                    return masks

        except Exception as e:
           print(f"Error removing background: {str(e)}")
           return None
    
    def postprocess_image(self, image, masks):
        if isinstance(masks, list):
            masks = [m.convert("L") for m in masks if isinstance(m, Image.Image)]
            mask = masks[0] if masks else None
        elif isinstance(mask, Image.Image):
            mask = mask.convert("L")

        mask_tensor = pil2tensor(mask)
        normalized_mask_tensor = torch.clamp(mask_tensor, 0, 1)

        orig_image = tensor2pil(image)
        orig_rgba = orig_image.convert("RGBA")
        r, g, b, _ = orig_rgba.split()
        res_image = Image.merge('RGBA', (r, g, b, mask))
        result = pil2tensor(res_image)

        return result, normalized_mask_tensor
        
    def process_image(self, image, model):
        try:
            images = self.preprocess_image(image, model)

            processed_images = []
            processed_masks = []
            
            for img in images:
                masks = self.remove_background(img, model)
                if not masks:
                    raise RuntimeError("Failed to generate masks")

                foreground, mask = self.postprocess_image(img, masks)

                processed_images.append(foreground)
                processed_masks.append(mask)

            if processed_images and processed_masks:
                return (torch.cat(processed_images, dim=0), torch.cat(processed_masks, dim=0))
            
            raise RuntimeError("No images processed")

        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None

# Node registration
NODE_CLASS_MAPPINGS = {
    "BackgroundBlur": BackgroundBlurNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BackgroundBlur": "Background Blur"
}
