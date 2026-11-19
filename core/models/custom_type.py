"""
Custom-type: 用户自定义等效电路模型
不经过 TLM 公式，直接解析并计算完整电路表达式
"""

from .base import TLMBaseModel
from ..component_parser import (
    parse, ParseError, compute_z_array, get_parameter_names, get_parameter_defaults
)
import numpy as np


class CustomModel(TLMBaseModel):
    def __init__(self, main_type='Custom', boundary='', zC_zero=False):
        super().__init__(main_type, boundary, zC_zero)
        self._custom_node = None

    def set_custom_expression(self, text):
        if text.strip():
            try:
                self._custom_node = parse(text)
            except ParseError:
                self._custom_node = None
        else:
            self._custom_node = None

    def get_all_parameter_names(self):
        if self._custom_node is None:
            return ['R0']
        params = get_parameter_names(self._custom_node)
        params.append('R0')
        return list(set(params))

    def get_parameter_defaults(self):
        d = {'R0': 0.5}
        if self._custom_node is not None:
            d.update(get_parameter_defaults(self._custom_node))
        return d

    def get_parameter_slots(self):
        if self._custom_node is None:
            return {}
        return {p: 'custom' for p in get_parameter_names(self._custom_node)}

    def compute_Z(self, f_arr, params):
        if self._custom_node is None:
            return np.full(len(f_arr), complex(params.get('R0', 0)),
                           dtype=np.complex128)
        Z = compute_z_array(self._custom_node, f_arr, params)
        return Z + params.get('R0', 0)
