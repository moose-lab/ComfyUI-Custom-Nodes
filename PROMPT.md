
# PROMPT

**This doc is for the users who wants to build a custom node for comfyui and how to build it automatically by llm.
Build what you want to build in comfyui workflow and everything is done by llm.**

***

## 1. normal node template
```
class CustomNodeName:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input1": ("INPUT_TYPE", 
                {
                    "default": value, 
                    "min": min, 
                    "max": max
                }),
                "input2": (["option1", "option2"], {
                    "default": "option1"
                    }),
            },
            "optional": {
                "optional_input": ("INPUT_TYPE", {
                    "default": None
                    })
            }
        }

    RETURN_TYPES = ("OUTPUT_TYPE",)
    RETURN_NAMES = ("output_name",)
    FUNCTION = "process"
    CATEGORY = "category/sub_category"
    OUTPUT_NODE = False # if you want to set node output, set it to True

    def process(self, input1, input2, optional_input=None):
        # your logic what you want to do
        return (output,)
```

## 2. Key of the component
 * INPUT_TYPES: input types
 * RETURN_TYPES: output types
 * CATEGORY: category and sub category
 * FUNCTION: process function
 
 2.1 INPUT_TYPES \
 **the type of the input:**
 ```markdown
    ("IMAGE",)        # image tensor [B, H, W, C] 
    ("LATENT",)       # latent space representation 
    ("MASK",)         # mask [H, W] 
    ("MODEL",)        # latent model 
    ("CLIP",)         # CLIP text encoder 
    ("CONDITIONING",) # comdition data 
    ("STRING", {"multiline": True}) # multiline text 
```
**the parameter of the input:**
```markdown
{
    "default": 0.5,             # default value
    "min": 0.0,                 # min value
    "max": 1.0,                 # max value 
    "step": 0.01,               # step value
    "tooltip": "threshold",     # tooltip
    "dynamicPrompts": True,     # dynamic prompts
    "file_upload": True         # file upload
}
```

2.2 RETURN_TYPES \
**the type of the output:**
```markdown
RETURN_TYPES = ("IMAGE", "LATENT", "STRING")
RETURN_NAMES = ("output image", "latent data", "output text")  # display name
```

2.3 CATEGORY \
**the category of the node:**
```markdown
"Image/Transform"     # image transform
"Latent/Composite"    # latent composite
"Conditioning/Filter" # conditioning filter
"Model/Loaders"       # model loaders
"Sampling"            # sampling strategy
"Utilities"           # utilities
"Custom"              # custom
```

2.4 FUNCTION \
**the function of the node:**
```markdown
FUNCTION = "process" # this is the function name
```

## 3. Supported node types template

3.1 Image node template
```python
class ImageEnhancement:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0})
            }
        }

        RETURN_TYPES = ("IMAGE", "MASK")
        RETURN_NAMES = ("processed_image", "alpha_mask")
        FUNCTION = "process"
        CATEGORY = "Image/Processing"
        OUTPUT_NODE = False

    def process(self, image, strength):
        # accelerate the process by gpu
        image = image.to(comfy.model_management.get_torch_device())
        enhanced = image * strength
        return (enhanced.clamp(0,1).cpu(),)

```

3.2 Node optimizer template
```python
class ModelOptimizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "optimize_level": (["fast", "balanced", "quality"],)
                "strength": ("FLOAT", {"default": 1.0})
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("optimized_model",)
    FUNCTION = "process"
    CATEGORY = "Model/Optimizers"
    OUTPUT_NODE = False

    def process(self, model, optimize_level):
        patched_model = model.clone()
        # inference code optimization
        if optimize_level == "fast":
            patched_model.model_options["optimized"] = True
        return (patched_model,)
```

3.3 Conditioning control template
```python
class AdvancedConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "control_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0})
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("advanced_conditioning",)
    FUNCTION = "process"
    CATEGORY = "Conditioning/Control"
    OUTPUT_NODE = False

    def process(self, conditioning, control_strength):
        new_conds = []
        for t in conditioning:
            n = [t[0], t[1].copy()]
            n[1]["control_strength"] = control_strength
            new_conds.append(n)
        return (new_conds,)
```

## 4. Advanced function

4.1 memory management
```python
# using the intermediate device to process
intermediate_device = comfy.model_management.intermediate_device()

# memory optimization mode
with torch.inference_mode():
    # inference code
    pass

# release the resource
del temporary_tensor
torch.cuda.empty_cache()
```

4.2 multiple devices sypport
```python
def process(self, image):
    device = comfy.model_management.get_torch_device()
    image = image.to(device)
    # inference code here
    return image.cpu()
```

4.3 progress bar callback
```python
def process(self, samples):
    with comfy.utils.ProgressBar(len(samples)) as progress:
        for sample in samples:
            # 处理逻辑
            progress.update(1)
```

4.4 validate the parameters
```python
@classmethod
def VALIDATE_INPUTS(cls, **kwargs):
    if kwargs.get("width") * kwargs.get("height") > 4096**2:
        return "the resolution is too high"
    return True
```

The bast practice advice:
* About the memory management, using `comfy.model_management` to manage memory, and using the `.cpu()` to release the gpu memory in a timely manner when processing large tensors.
* compatibility handling
```python
    if hasattr(comfy.samplers, 'new_feature'):
        # use the new feature
        pass
    else: 
        # use the old feature
        pass
```
* performance optimization
using `@torch.no_grad()` to decorate the inference code to disable gradient calculation and reduce memory usage, and using iin-place operations asap.
* error handling
using `try-except` to handle the error and return the error message.

## 5. Debugging and test advice

5.1 Preview the intermediate results
```python
class PreviewNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"data": ("*",)}}
    
    OUTPUT_NODE = True
    FUNCTION = "show"
    
    def show(self, data):
        print(f"Debug Shape: {data.shape if hasattr(data,'shape') else str(data)}")
        return {"ui": {"preview": data}}
```

5.2 unit test mode
```python
    def unit_test_0():
        test_input = torch.randn(1, 512, 512, 3)
        node = ImageEnhancement()
        output = node.process(test_input, 1.5)
        assert output[0].shape == test_input.shape  
```

## 6. Release rules

6.1 directory structure
```markdown
confyui-custom-nodes/
└── nodes/
    ├── image_processors.py
    ├── model_tools.py
    ├── background_blur_node.py # example node
    └── js/
        └── widgets.js  # UI
```
The development of custom nodes is generally defined in the nodes directory, and the UI is placed in the js directory. 
The registration is placed in the __init__.py outside the nodes directory.

6.2 metadata notation
```markdown
Author: Your Name
Description: the comfyui custom node
Version: 1.0.0
Compatibility: ComfyUI 1.2+
Dependencies: torchvision>=0.15
```