"""
字符串解析器：将组件定义字符串解析为阻抗计算 AST
语法:
  串联: A-B  表示 A 与 B 串联
  并联: (AB) 表示 A 与 B 并联 (括号内所有元素)
  嵌套: (A(BC)) 表示 A 与 (B//C) 并联
示例:
  R1-(R2W1)  →  Series(Component(R1), Parallel(Component(R2), Component(W1)))
  (R1C1)     →  Parallel(Component(R1), Component(C1))
  R1-R2-R3   →  Series(Component(R1), Component(R2), Component(R3))
"""

from dataclasses import dataclass, field
from typing import Union
import re
import numpy as np
from .component_impedance import (
    z_havriliak_negami, z_havriliak_negami_alt,
    z_gerischer, z_gerischer_alt,
    z_warburg_reflective, z_warburg_transmissive,
)


COMPONENT_RE = re.compile(r'^([A-Za-z_]\w*)$')
COMPONENT_TYPE_RE = re.compile(r'^([A-Za-z]+)\d*$')

VALID_COMPONENT_TYPES = {'R', 'C', 'L', 'Wo', 'Ws', 'W', 'H', 'Ha', 'G', 'Ga'}


def validate_types(node) -> None:
    if isinstance(node, ComponentNode):
        if node.ctype not in VALID_COMPONENT_TYPES:
            raise ParseError(f"Unknown component type: '{node.ctype}' in '{node.name}'")
        return
    for child in node.children:
        validate_types(child)


class ParseError(Exception):
    pass


def _get_component_type(name: str) -> str:
    m = COMPONENT_TYPE_RE.match(name)
    if m:
        return m.group(1)
    return name


@dataclass
class ComponentNode:
    name: str
    ctype: str = field(init=False)

    def __post_init__(self):
        self.ctype = _get_component_type(self.name)

    def __repr__(self):
        return f"Component({self.name})"


@dataclass
class SeriesNode:
    children: list = field(default_factory=list)

    def __repr__(self):
        return f"Series({self.children})"


@dataclass
class ParallelNode:
    children: list = field(default_factory=list)

    def __repr__(self):
        return f"Parallel({self.children})"


def _tokenize(text: str) -> list:
    tokens = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in '()-':
            tokens.append(ch)
            i += 1
        elif ch.isspace():
            i += 1
        else:
            j = i
            while j < len(text) and text[j] not in '()-' and not text[j].isspace():
                j += 1
            # Split contiguous word at digit→letter boundaries
            # e.g. R1R2 → R1,R2   R2Zw → R2,Zw   C1 → C1
            sub_start = i
            for k in range(i + 1, j):
                if text[k - 1].isdigit() and text[k].isalpha():
                    sub = text[sub_start:k]
                    if not COMPONENT_RE.match(sub):
                        raise ParseError(
                            f"Invalid component name: '{sub}' in '{text[i:j]}'")
                    tokens.append(sub)
                    sub_start = k
            sub = text[sub_start:j]
            if not COMPONENT_RE.match(sub):
                raise ParseError(f"Invalid component name: '{sub}'")
            tokens.append(sub)
            i = j
    return tokens


class Parser:
    """
    Recursive descent parser for impedance expression grammar.

    Grammar:
      expression = series
      series     = parallel ('-' parallel)*
      parallel   = '(' series ')' | component
    """

    def __init__(self, tokens: list):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Union[str, None]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected=None):
        tok = self.peek()
        if tok is None:
            raise ParseError("Unexpected end of expression")
        if expected is not None and tok != expected:
            raise ParseError(f"Expected '{expected}', got '{tok}'")
        self.pos += 1
        return tok

    def parse(self):
        node = self._parse_series()
        if self.peek() is not None:
            raise ParseError(f"Unexpected token '{self.peek()}' at end")
        return node

    def _parse_series(self):
        left = self._parse_parallel()
        while self.peek() == '-':
            self.consume('-')
            right = self._parse_parallel()
            if not isinstance(left, SeriesNode):
                left = SeriesNode(children=[left])
            left.children.append(right)
        return left

    def _parse_parallel(self):
        if self.peek() == '(':
            # Parse: '(' series ')'
            # Then collect additional items inside the parens as parallel siblings
            self.consume('(')
            items = [self._parse_series()]
            while self.peek() and self.peek() not in ')-':
                items.append(self._parse_series())
            self.consume(')')
            if len(items) == 1:
                return items[0]
            return ParallelNode(children=items)
        else:
            return self._parse_component()

    def _parse_component(self):
        tok = self.consume()
        return ComponentNode(name=tok)


def parse(text: str):
    tokens = _tokenize(text)
    if not tokens:
        raise ParseError("Empty expression")
    parser = Parser(tokens)
    return parser.parse()


def collect_component_names(node) -> list:
    if isinstance(node, ComponentNode):
        return [node.name]
    names = []
    for child in node.children:
        names.extend(collect_component_names(child))
    return names


def collect_component_types(node) -> dict:
    if isinstance(node, ComponentNode):
        return {node.name: node.ctype}
    types = {}
    for child in node.children:
        types.update(collect_component_types(child))
    return types


def _safe_coth(x):
    ax = np.abs(x)
    if ax < 1e-12:
        return 1.0 / x + x / 3.0
    if np.real(x) > 50:
        return 1.0 + 2.0 * np.exp(-2.0 * x)
    ep = np.exp(x)
    en = np.exp(-x)
    return (ep + en) / (ep - en)


def compute_z(node, f, params: dict) -> np.complex128:
    omega = 2.0 * np.pi * f
    if isinstance(node, ComponentNode):
        ctype = node.ctype
        if ctype == 'R':
            return np.complex128(params[f'{node.name}_R'] + 0j)
        elif ctype == 'C':
            C = params[f'{node.name}_C']
            p = params[f'{node.name}_p']
            return np.complex128(1.0 / (C * (1j * omega) ** p))
        elif ctype == 'L':
            L = params[f'{node.name}_L']
            return np.complex128(1j * omega * L)
        elif ctype == 'Wo':
            Rd = params[f'{node.name}_Rd']
            tau = params[f'{node.name}_tau']
            p = params[f'{node.name}_p']
            return z_warburg_reflective(Rd, tau, p, omega)
        elif ctype == 'Ws':
            Rd = params[f'{node.name}_Rd']
            tau = params[f'{node.name}_tau']
            p = params[f'{node.name}_p']
            return z_warburg_transmissive(Rd, tau, p, omega)
        elif ctype == 'W':
            Aw = params[f'{node.name}_Aw']
            return Aw / np.sqrt(1j * omega)
        elif ctype == 'H':
            dC = params[f'{node.name}_dC']; tau = params[f'{node.name}_tau']
            a = params[f'{node.name}_a']; b = params[f'{node.name}_b']
            return z_havriliak_negami(dC, tau, a, b, omega)
        elif ctype == 'Ha':
            R = params[f'{node.name}_R']; tau = params[f'{node.name}_tau']
            a = params[f'{node.name}_a']; b = params[f'{node.name}_b']
            return z_havriliak_negami_alt(R, tau, a, b, omega)
        elif ctype == 'G':
            Y = params[f'{node.name}_Y']; k = params[f'{node.name}_k']
            n = params[f'{node.name}_n']
            return z_gerischer(Y, k, n, omega)
        elif ctype == 'Ga':
            R = params[f'{node.name}_R']; tau = params[f'{node.name}_tau']
            n = params[f'{node.name}_n']
            return z_gerischer_alt(R, tau, n, omega)
        raise ValueError(f"Unknown component type: {ctype}")
    elif isinstance(node, SeriesNode):
        z = np.complex128(0)
        for child in node.children:
            z += compute_z(child, f, params)
        return z
    elif isinstance(node, ParallelNode):
        z_inv = np.complex128(0)
        for child in node.children:
            z_inv += 1.0 / compute_z(child, f, params)
        return 1.0 / z_inv
    raise ValueError(f"Unknown node type: {type(node)}")


def compute_z_array(node, f_arr: np.ndarray, params: dict) -> np.ndarray:
    return np.array([compute_z(node, f, params) for f in f_arr])


def get_parameter_names(node) -> list:
    if isinstance(node, ComponentNode):
        ctype = node.ctype
        base = node.name
        if ctype == 'R':
            return [f'{base}_R']
        elif ctype == 'C':
            return [f'{base}_C', f'{base}_p']
        elif ctype == 'L':
            return [f'{base}_L']
        elif ctype in ('Wo', 'Ws'):
            return [f'{base}_Rd', f'{base}_tau', f'{base}_p']
        elif ctype == 'W':
            return [f'{base}_Aw']
        elif ctype == 'H':
            return [f'{base}_dC', f'{base}_tau', f'{base}_a', f'{base}_b']
        elif ctype == 'Ha':
            return [f'{base}_R', f'{base}_tau', f'{base}_a', f'{base}_b']
        elif ctype == 'G':
            return [f'{base}_Y', f'{base}_k', f'{base}_n']
        elif ctype == 'Ga':
            return [f'{base}_R', f'{base}_tau', f'{base}_n']
        return []
    names = []
    for child in node.children:
        names.extend(get_parameter_names(child))
    return names


def get_parameter_defaults(node) -> dict:
    if isinstance(node, ComponentNode):
        ctype = node.ctype
        base = node.name
        if ctype == 'R':
            return {f'{base}_R': 1.0}
        elif ctype == 'C':
            return {f'{base}_C': 1e-3, f'{base}_p': 0.8}
        elif ctype == 'L':
            return {f'{base}_L': 1e-6}
        elif ctype in ('Wo', 'Ws'):
            return {f'{base}_Rd': 0.01, f'{base}_tau': 10, f'{base}_p': 0.5}
        elif ctype == 'W':
            return {f'{base}_Aw': 1.0}
        elif ctype == 'H':
            return {f'{base}_dC': 1e-6, f'{base}_tau': 1.0, f'{base}_a': 0.9, f'{base}_b': 0.9}
        elif ctype == 'Ha':
            return {f'{base}_R': 1.0, f'{base}_tau': 1.0, f'{base}_a': 0.7, f'{base}_b': 0.8}
        elif ctype == 'G':
            return {f'{base}_Y': 1.0, f'{base}_k': 1.0, f'{base}_n': 0.5}
        elif ctype == 'Ga':
            return {f'{base}_R': 1.0, f'{base}_tau': 1.0, f'{base}_n': 0.5}
        return {}
    defaults = {}
    for child in node.children:
        defaults.update(get_parameter_defaults(child))
    return defaults
