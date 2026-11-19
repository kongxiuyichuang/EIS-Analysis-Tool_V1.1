from .base import TLMBaseModel, MAIN_MODELS, BOUNDARY_CONDITIONS, ZC_ZERO_SUPPORT, PRESET_TEMPLATES, TLMComponentNames
from .z_type import ZTypeModel
from .t_type import TTypeModel
from .e_type import ETypeModel
from .universal_type import UniversalTypeModel
from .custom_type import CustomModel


MODEL_CLASSES = {
    'Z-type': ZTypeModel,
    'T-type': TTypeModel,
    'E-type': ETypeModel,
    'Universal-type': UniversalTypeModel,
    'Custom': CustomModel,
}


def create_model(main_type: str, boundary: str, zC_zero: bool = False) -> TLMBaseModel:
    cls = MODEL_CLASSES.get(main_type)
    if cls is None:
        raise ValueError(f"Unknown model type: {main_type}")
    return cls(main_type=main_type, boundary=boundary, zC_zero=zC_zero)
