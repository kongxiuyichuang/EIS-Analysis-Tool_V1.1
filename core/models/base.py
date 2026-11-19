"""
模型基类：定义所有 TLM 模型的统一接口
"""

from abc import ABC, abstractmethod
import numpy as np
from core.component_parser import (
    parse, ParseError, compute_z_array, collect_component_names,
    get_parameter_names, get_parameter_defaults, ComponentNode,
    SeriesNode, ParallelNode
)


MAIN_MODELS = ['Z-type', 'T-type', 'E-type', 'Universal-type', 'Custom']

BOUNDARY_CONDITIONS = {
    'Z-type':     ['P-Q', 'open-Q', 'open-open'],
    'T-type':     ['P-Q', 'P-P symmetric', 'open-Q', 'open-open'],
    'E-type':     ['P-Q', 'open-Q', 'open-open'],
    'Universal-type': ['general'],
    'Custom':     ['general'],
}

ZC_ZERO_SUPPORT = {
    'Z-type': ['P-Q', 'open-Q', 'open-open'],
    'T-type': ['P-Q', 'P-P symmetric', 'open-Q', 'open-open'],
    'E-type': [],
    'Universal-type': [],
    'Custom': [],
}

PRESET_TEMPLATES = {
    'non_faradaic': {
        'zA': 'R1',
        'zB': 'C1',
        'zC': '',
        'zP': 'R2',
        'zQ': 'R2',
        'ZV': '',
        'ZW': '',
        'ZX': '',
        'ZY': '',
    },
    'faradaic': {
        'zA': 'R1',
        'zB': '(R2C2)',
        'zC': 'R3',
        'zP': 'R4',
        'zQ': 'R5',
        'ZV': '',
        'ZW': '',
        'ZX': '',
        'ZY': '',
    },
}

TLMComponentNames = ['zA', 'zB', 'zC', 'zP', 'zQ', 'ZV', 'ZW', 'ZX', 'ZY']


class TLMBaseModel(ABC):
    def __init__(self, main_type: str, boundary: str, zC_zero: bool = False):
        self.main_type = main_type
        self.boundary = boundary
        self.zC_zero = zC_zero
        self._zA_node = None
        self._zB_node = None
        self._zC_node = None
        self._zP_node = None
        self._zQ_node = None
        self._ZV_node = None
        self._ZW_node = None
        self._ZX_node = None
        self._ZY_node = None

    def set_component(self, name: str, text: str):
        if text.strip():
            try:
                node = parse(text)
            except ParseError:
                node = None
        else:
            node = None
        setattr(self, f'_{name}_node', node)

    def _compute_sub_z(self, name: str, f_arr: np.ndarray, params: dict) -> np.ndarray:
        node = getattr(self, f'_{name}_node', None)
        if node is None:
            return np.zeros(len(f_arr), dtype=np.complex128)
        return compute_z_array(node, f_arr, params)

    def get_all_component_names(self) -> list:
        names = []
        for attr in ['_zA_node', '_zB_node', '_zC_node', '_zP_node', '_zQ_node',
                     '_ZV_node', '_ZW_node', '_ZX_node', '_ZY_node']:
            node = getattr(self, attr, None)
            if node is not None:
                names.extend(collect_component_names(node))
        return list(set(names))

    def get_all_parameter_names(self) -> list:
        params = []
        for attr in ['_zA_node', '_zB_node', '_zC_node', '_zP_node', '_zQ_node',
                     '_ZV_node', '_ZW_node', '_ZX_node', '_ZY_node']:
            node = getattr(self, attr, None)
            if node is not None:
                params.extend(get_parameter_names(node))
        params.extend(['R0', 'L'])
        return list(set(params))

    def get_parameter_defaults(self) -> dict:
        defaults = {}
        for attr in ['_zA_node', '_zB_node', '_zC_node', '_zP_node', '_zQ_node',
                     '_ZV_node', '_ZW_node', '_ZX_node', '_ZY_node']:
            node = getattr(self, attr, None)
            if node is not None:
                defaults.update(get_parameter_defaults(node))
        defaults['R0'] = 0.5
        defaults['L'] = 0.01
        return defaults

    def get_parameter_slots(self) -> dict:
        slot_map = {}
        for attr, slot in [
            ('_zA_node', 'zA'), ('_zC_node', 'zC'),
            ('_zB_node', 'zB'),
            ('_zP_node', 'zP'), ('_zQ_node', 'zQ'),
            ('_ZV_node', 'ZV'), ('_ZW_node', 'ZW'),
            ('_ZX_node', 'ZX'), ('_ZY_node', 'ZY'),
        ]:
            node = getattr(self, attr, None)
            if node is not None:
                for p in get_parameter_names(node):
                    slot_map[p] = slot
        return slot_map

    @abstractmethod
    def compute_Z(self, f_arr: np.ndarray, params: dict) -> np.ndarray:
        pass
