"""Adapted pyrecodes utility helpers.

Source: pyrecodes, BSD-3-Clause license. This local copy keeps the JSON and
small utility functions needed by the household-agent workflow; geometry
helpers lazily import optional dependencies. See ../../THIRD_PARTY_NOTICES.md.
"""

import json
import importlib
from typing import Union


def _get_shapely():
    try:
        import shapely
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "shapely is required for pyrecodes geometry utilities, but it is not "
            "needed for the household survey agent workflow."
        ) from exc
    return shapely

def read_json_file(file_name: str) -> dict:
    """Reads a JSON file and returns its contents as a dictionary.

    Args:
        file_name (str): The name of the JSON file to be read.

    Returns:
        dict: The dictionary representation of the JSON data.
    """
    with open(file_name, 'r') as file:
        return json.load(file)

def resolve_folder_paths(obj: Union[dict, list, str], folder_name: str) -> Union[dict, list, str]:
    """Recursively replaces the ``{folder}`` placeholder in all string values with ``folder_name``.

    Use ``{folder}`` in system configuration JSON to reference files relative to the
    input folder without hardcoding its name.

    Args:
        obj: A dict, list, or string parsed from JSON.
        folder_name: The folder name passed to ``main.run()``.

    Returns:
        The same structure with every ``{folder}`` occurrence replaced.
    """
    if isinstance(obj, dict):
        return {k: resolve_folder_paths(v, folder_name) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_folder_paths(item, folder_name) for item in obj]
    if isinstance(obj, str):
        return obj.replace('{folder}', folder_name)
    return obj

def get_class(module_name: str, class_name: str, folder_name: str) -> object:
    """Imports a class from a file"""
    class_module = importlib.import_module(f'pyrecodes.{folder_name}.{module_name}')
    target_class = getattr(class_module, class_name)
    return target_class

def create_locality_polygon(bounding_box: list):
    shapely = _get_shapely()
    return shapely.Polygon([(lat, long) for long, lat in bounding_box])

def component_inside_bounding_box(component_geometry, polygon) -> bool:
    return component_geometry.within(polygon)

def create_component_geometry_from_wkt(geometry_string: str):
    shapely = _get_shapely()
    return shapely.wkt.loads(geometry_string)

def create_component_geometry_as_point(component_location: list):
    shapely = _get_shapely()
    return shapely.Point(component_location[0], component_location[1])

def get_locality_coordinates_from_geojson(locality_info: dict) -> dict:
    geojson_file = read_json_file(locality_info['GeoJSON']['Filename'])
    return {'BoundingBox': geojson_file['features'][0]['geometry']['coordinates'][0][0]}

def format_locality_id(locality_string) -> int:
    return int(locality_string.split(' ')[-1])

def json_deepcopy(input: Union[list, dict]) -> Union[list, dict]:
    """Fast deep copy for nested dicts/lists of JSON-serializable primitives."""
    return json.loads(json.dumps(input))

def unpack_time_stepping_rules(time_stepping_rules: list) -> list:
    """
    Unpacks time stepping rules into a list of time steps.
    """
    time_steps = []
    for time_stepping in time_stepping_rules:
        time_steps += list(range(time_stepping['start'], time_stepping['end'], time_stepping['step']))
    return time_steps
