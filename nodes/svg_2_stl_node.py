# graph TD
#    A [read svg file] --> B[parsing the path data]
#    B --> C[calculate the points]
#    C --> D[construct the wall]
#    D --> E[generate the mesh]
#    E --> F[stl output]
import os
import gmsh
import numpy as np
from svg.path import parse_path
from svg.path import Close, CubicBezier, Line, Move
from xml.dom import minidom
import pathlib


class SVG2STLNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "svg_path": ("STRING", {
                    "default": "path/to/your.svg",
                    "multiline": False,
                }),
                "thickness": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.1
                }),
                "definition": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "skip": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10,
                    "step": 1
                }),
                "show": ("BOOLEAN", {
                    "default": False,
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("stl_path",)
    FUNCTION = "svg_2_stl"
    CATEGORY = "3d/svg"

    @staticmethod
    def parse_svg_into_steps(path: str) -> list:
        """parsing the svg path data"""
        try:
            doc = minidom.parse(path)
            paths = doc.getElementsByTagName("path")
            if not paths:
                raise ValueError("No path found in SVG file")
            path_str = paths[0].getAttribute("d")
            doc.unlink()
            return parse_path(path_str)
        except Exception as e:
            raise ValueError(f"Error parsing SVG file: {str(e)}")
    
    def svg_2_stl(self, svg_path: str, thickness: float, definition: int, skip: int, show: bool):
        """convert svg to stl"""
        if not os.path.exists(svg_path):
            raise FileNotFoundError(f"SVG file not found: {svg_path}")

        try:
            # parsing the svg path data
            steps = self.parse_svg_into_steps(svg_path)

            # calculate the points
            shapes = []
            shape = []

            for step in steps:
                if isinstance(step, Line):
                    shape.append([step.start.real, step.start.imag])
                
                elif isinstance(step, Close):
                    if shape:  # only add when shape is not empty
                        shapes.append(shape)
                        shape = []
                
                elif not isinstance(step, Move):
                    for t in np.linspace(0, 1, definition, endpoint=False):
                        p = step.point(t)
                        shape.append([p.real, p.imag])

            # if the last shape is not added (no Close command), add it
            if shape:
                shapes.append(shape)

            if not shapes:
                raise ValueError("No valid shapes found in SVG")

            # calculate the boundary and add padding
            points = np.vstack(shapes)
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            x_pad = 0.1 * (x_max - x_min)
            y_pad = 0.1 * (y_max - y_min)
            corners = [
                [x_min - x_pad, y_min - y_pad],
                [x_min - x_pad, y_max + y_pad],
                [x_max + x_pad, y_max + y_pad],
                [x_max + x_pad, y_min - y_pad],
            ]
            shapes.append(corners)

            # build the 3d model
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 1)
            gmsh.model.add("svg_model")

            z_floor = 0
            z_ceiling = thickness

            factory = gmsh.model.geo
            floor_lines = []
            ceiling_lines = []
            wall_lines = []

            # build the walls
            for shape in shapes[skip:]:
                floor_lines.append([])
                floor_points = [factory.addPoint(*shape[0], z_floor)]
                for vertex in shape[1:]:
                    floor_points.append(factory.addPoint(*vertex, z_floor))
                    floor_lines[-1].append(factory.addLine(floor_points[-2], floor_points[-1]))
                floor_lines[-1].append(factory.addLine(floor_points[-1], floor_points[0]))

                ceiling_lines.append([])
                ceiling_points = [factory.addPoint(*shape[0], z_ceiling)]
                for vertex in shape[1:]:
                    ceiling_points.append(factory.addPoint(*vertex, z_ceiling))
                    ceiling_lines[-1].append(
                        factory.addLine(ceiling_points[-2], ceiling_points[-1])
                    )
                ceiling_lines[-1].append(factory.addLine(ceiling_points[-1], ceiling_points[0]))

                wall_lines.append([])
                for floor_point, ceiling_point in zip(floor_points, ceiling_points):
                    wall_line = factory.addLine(floor_point, ceiling_point)
                    wall_lines[-1].append(wall_line)

                # create the side walls
                for i in range(1, len(floor_lines[-1])):
                    wall = factory.addCurveLoop(
                        [
                            floor_lines[-1][i - 1],
                            wall_lines[-1][i],
                            -ceiling_lines[-1][i - 1],
                            -wall_lines[-1][i - 1],
                        ]
                    )
                    factory.addPlaneSurface([wall])
                
                # connect the last face
                wall = factory.addCurveLoop(
                    [
                        floor_lines[-1][-1],
                        wall_lines[-1][0],
                        -ceiling_lines[-1][-1],
                        -wall_lines[-1][-1],
                    ]
                )
                factory.addPlaneSurface([wall])

            # build the floor and ceiling
            floor = []
            for lines in floor_lines:
                hole = factory.addCurveLoop(lines)
                floor.append(hole)
            factory.addPlaneSurface(floor)

            ceiling = []
            for lines in ceiling_lines:
                hole = factory.addCurveLoop(lines)
                ceiling.append(hole)
            factory.addPlaneSurface(ceiling)

            # generate the mesh
            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate()

            # ensure the output directory exists
            output_path = str(pathlib.Path(svg_path).with_suffix(".stl"))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # save the stl file
            gmsh.write(output_path)

            if show:
                gmsh.fltk.run()

            gmsh.finalize()
            
            return (output_path,)

        except Exception as e:
            if gmsh:
                gmsh.finalize()
            raise Exception(f"Error converting SVG to STL: {str(e)}")

# Node registration
NODE_CLASS_MAPPINGS = {
    "SVG2STL": SVG2STLNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SVG2STL": "SVG to STL"
}

