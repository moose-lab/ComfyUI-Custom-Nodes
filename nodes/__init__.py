import os
import importlib.util

def load_nodes_from_directory():
    nodes_directory = os.path.dirname(os.path.abspath(__file__))
    node_class_mappings = {}
    node_display_name_mappings = {}

    # Iterate through all Python files in the nodes directory
    for filename in os.listdir(nodes_directory):
        if filename.endswith('.py') and filename != '__init__.py':
            module_name = filename[:-3]  # Remove .py extension
            module_path = os.path.join(nodes_directory, filename)

            # Load the module dynamically
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Update mappings if they exist in the module
            if hasattr(module, 'NODE_CLASS_MAPPINGS'):
                node_class_mappings.update(module.NODE_CLASS_MAPPINGS)
            if hasattr(module, 'NODE_DISPLAY_NAME_MAPPINGS'):
                node_display_name_mappings.update(module.NODE_DISPLAY_NAME_MAPPINGS)

    return node_class_mappings, node_display_name_mappings

# Load all nodes and create the mappings
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = load_nodes_from_directory()