from .nodes.svg_2_stl_node import NODE_CLASS_MAPPINGS as SVG_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as SVG_DISPLAY
from .nodes.background_blur_node import NODE_CLASS_MAPPINGS as BLUR_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BLUR_DISPLAY
from .nodes.image_blend_node import NODE_CLASS_MAPPINGS as BLEND_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as BLEND_DISPLAY


NODE_CLASS_MAPPINGS = {
    **SVG_MAPPINGS,
    **BLUR_MAPPINGS,
    **BLEND_MAPPINGS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **SVG_DISPLAY,
    **BLUR_DISPLAY,
    **BLEND_DISPLAY
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']