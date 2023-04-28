from enum import Enum
# from nerfstudio.field_components.field_heads import FieldHeadNames

class RENINeuSFieldHeadNames(Enum):
    """Possible field outputs"""
    ALBEDO = "albedo"
    VISIBILITY = "visibility"