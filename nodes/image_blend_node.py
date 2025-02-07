import torch
import comfy.model_management
from torchvision.transforms.functional import resize
from comfy.utils import ProgressBar

class ImageBlend:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "IMAGE1": ("IMAGE",),
                "IMAGE2": ("IMAGE",),
                "blend_ratio": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Blend ratio (0.0=IMAGE1, 1.0=IMAGE2)"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_image",)
    FUNCTION = "blend_images"
    CATEGORY = "Image/Composite"
    OUTPUT_NODE = False

    @torch.no_grad()
    def blend_images(self, image1, image2, blend_ratio):
        # validate the inputs
        if image1 is None or image2 is None:
            raise ValueError("input image is null")
        
        # move to the gpu
        device = comfy.model_management.get_torch_device()
        image1 = image1.to(device)
        image2 = image2.to(device)

        # get the maximum size
        max_h = max(image1.shape[1], image2.shape[1])
        max_w = max(image1.shape[2], image2.shape[2])
        
        # using the resize function to resize the image: bio-linear interpolation
        if image1.shape[1:3] != (max_h, max_w):
            image1 = resize(image1.permute(0,3,1,2), [max_h, max_w], 
                          interpolation=torch.nn.functional.interpolate).permute(0,2,3,1)
        
        if image2.shape[1:3] != (max_h, max_w):
            image2 = resize(image2.permute(0,3,1,2), [max_h, max_w],
                          interpolation=torch.nn.functional.interpolate).permute(0,2,3,1)

        # blend computing
        blended = (1 - blend_ratio) * image1 + blend_ratio * image2
        
        # limit the range of the output image
        blended = torch.clamp(blended, 0.0, 1.0)
        
        # move to the cpu
        return (blended.cpu(),)

    @classmethod
    def VALIDATE_INPUTS(cls, IMAGE1, IMAGE2, **kwargs):
        if IMAGE1 is None or IMAGE2 is None:
            return "input image is null"
        if IMAGE1.shape[0] != IMAGE2.shape[0]:
            return "input image batch size is not equal"
        return True

NODE_CLASS_MAPPINGS = {
    "ImageBlend": ImageBlend
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageBlend": "🔀 Image Blend"
} 