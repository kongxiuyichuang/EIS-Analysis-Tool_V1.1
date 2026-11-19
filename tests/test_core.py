#!/usr/bin/env python3
"""Core module tests for TLM V2.0"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

# Test component parser
from core.component_parser import parse, parse as _p
from core.component_parser import (
    ComponentNode, SeriesNode, ParallelNode,
    compute_z, collect_component_names, get_parameter_names,
    get_parameter_defaults, ParseError
)


def test_parser_single_component():
    node = parse('R1')
    assert isinstance(node, ComponentNode)
    assert node.name == 'R1'
    assert node.ctype == 'R'
    print("  ✓ Single component: R1")


def test_parser_series():
    node = parse('R1-R2')
    assert isinstance(node, SeriesNode)
    assert len(node.children) == 2
    assert node.children[0].name == 'R1'
    assert node.children[1].name == 'R2'
    print("  ✓ Series: R1-R2")


def test_parser_parallel():
    node = parse('(R1R2)')
    assert isinstance(node, ParallelNode)
    assert len(node.children) == 2
    print("  ✓ Parallel: (R1R2)")


def test_parser_complex():
    node = parse('R1-(R2Wo1)')
    assert isinstance(node, SeriesNode)
    assert len(node.children) == 2
    assert isinstance(node.children[1], ParallelNode)
    print("  ✓ Complex: R1-(R2Wo1)")


def test_collect_names():
    node = parse('R1-(R2Wo1)-C1')
    names = collect_component_names(node)
    assert 'R1' in names
    assert 'R2' in names
    assert 'Wo1' in names
    assert 'C1' in names
    print("  ✓ Collect names")


def test_get_params():
    node = parse('R1-(R2Wo1)-C1')
    params = get_parameter_names(node)
    assert 'R1_R' in params
    assert 'R2_R' in params
    assert 'Wo1_Rd' in params
    assert 'Wo1_tau' in params
    assert 'C1_C' in params
    assert 'C1_p' in params
    print("  ✓ Get parameter names")


def test_defaults():
    node = parse('R1-C1')
    defaults = get_parameter_defaults(node)
    assert 'R1_R' in defaults
    assert defaults['R1_R'] == 1.0
    assert 'C1_C' in defaults
    assert defaults['C1_C'] == 1e-3
    assert 'C1_p' in defaults
    assert defaults['C1_p'] == 0.8
    print("  ✓ Default values")


def test_compute_simple():
    node = parse('R1')
    params = {'R1_R': 100.0}
    z = compute_z(node, 1000, params)
    assert abs(z - 100.0) < 1e-10
    print("  ✓ Compute simple resistor")

    node = parse('R1-R2')
    params = {'R1_R': 100.0, 'R2_R': 50.0}
    z = compute_z(node, 1000, params)
    assert abs(z - 150.0) < 1e-10
    print("  ✓ Compute series")

    node = parse('(R1R2)')
    params = {'R1_R': 100.0, 'R2_R': 100.0}
    z = compute_z(node, 1000, params)
    assert abs(z - 50.0) < 1e-10
    print("  ✓ Compute parallel")


def test_series_then_parallel():
    node = parse('(R1-R2C1)')
    assert isinstance(node, ParallelNode)
    assert len(node.children) == 2
    assert isinstance(node.children[0], SeriesNode)
    assert isinstance(node.children[1], ComponentNode)
    params = {'R1_R': 100.0, 'R2_R': 200.0, 'C1_C': 1e-6, 'C1_p': 1.0}
    z = compute_z(node, 1000, params)
    assert np.isfinite(z)
    assert abs(z) > 0
    print(f"  ✓ Series-then-parallel: (R1-R2C1) = {z:.4f} Ω")


def test_parse_error():
    try:
        parse('123invalid')
        assert False, "Should have raised ParseError"
    except ParseError:
        pass
    print("  ✓ Parse error handling")


# Test models
from core.models import create_model
from core.models.base import MAIN_MODELS, BOUNDARY_CONDITIONS


def test_models_create():
    for mt in MAIN_MODELS:
        boundaries = BOUNDARY_CONDITIONS[mt]
        for bc in boundaries:
            model = create_model(mt, bc)
            assert model is not None
            assert model.main_type == mt
            assert model.boundary == bc
    print("  ✓ All models created successfully")


def test_z_type_non_faradaic():
    model = create_model('Z-type', 'open-open', zC_zero=True)
    model.set_component('zA', 'R1')
    model.set_component('zB', 'C1')
    model.set_component('zC', '')
    model.set_component('zP', '')
    model.set_component('zQ', '')
    f_arr = np.array([1000, 100, 10])
    params = {'R1_R': 1.0, 'C1_C': 1e-3, 'C1_p': 0.8, 'R0': 0.5, 'L': 0.01}
    Z = model.compute_Z(f_arr, params)
    assert len(Z) == 3
    assert np.all(np.isfinite(Z))
    print(f"  ✓ Z-type open-open zC=0 (Non-Faradaic): Z={Z}")


def test_e_type():
    model = create_model('E-type', 'open-open')
    model.set_component('zA', 'R1')
    model.set_component('zB', 'C1')
    model.set_component('zC', 'R2')
    f_arr = np.array([1000, 100, 10])
    params = {'R1_R': 1.0, 'R2_R': 1.0, 'C1_C': 1e-3, 'C1_p': 0.8, 'R0': 0.5, 'L': 0.01}
    Z = model.compute_Z(f_arr, params)
    assert len(Z) == 3
    assert np.all(np.isfinite(Z))
    print(f"  ✓ E-type open-open: Z={Z}")


def test_universal_type():
    model = create_model('Universal-type', 'general')
    model.set_component('zA', 'R1')
    model.set_component('zB', 'C1')
    model.set_component('zC', 'R2')
    model.set_component('zP', 'R3')
    model.set_component('zQ', 'R3')
    model.set_component('ZV', 'R4')
    model.set_component('ZW', 'R4')
    model.set_component('ZX', 'R4')
    model.set_component('ZY', 'R4')
    f_arr = np.array([1000, 100])
    params = {
        'R1_R': 1.0, 'R2_R': 1.0, 'R3_R': 100.0, 'R4_R': 100.0,
        'C1_C': 1e-3, 'C1_p': 0.8,
        'R0': 0.5, 'L': 0.01
    }
    Z = model.compute_Z(f_arr, params)
    assert len(Z) == 2
    assert np.all(np.isfinite(Z))
    print(f"  ✓ Universal-type: Z={Z}")


# Test fitting engine
from core.fitting_engine import FittingEngine, FitParams


def test_fit_simple():
    model = create_model('Z-type', 'open-open', zC_zero=True)
    model.set_component('zA', 'R1')
    model.set_component('zB', 'C1')
    model.set_component('zC', '')

    f_arr = np.logspace(3, -1, 20)
    true_params = {'R1_R': 5.0, 'C1_C': 2e-3, 'C1_p': 0.85, 'R0': 1.0, 'L': 0.005}
    z_true = model.compute_Z(f_arr, true_params)
    z_noisy = z_true + 0.01 * np.random.randn(len(z_true)) + 0.01j * np.random.randn(len(z_true))

    fit_params = FitParams(
        names=['R1_R', 'C1_C', 'C1_p', 'R0', 'L'],
        initial=[1.0, 1e-3, 0.5, 0.1, 0.001],
        lower=[1e-3, 1e-9, 0.1, 1e-4, 1e-6],
        upper=[100, 1.0, 1.0, 10.0, 1.0],
        fixed=[False, False, False, False, False],
    )
    engine = FittingEngine(model, f_arr, z_noisy)
    result = engine.run(fit_params)
    assert result.success
    assert np.all(np.isfinite(result.z_fit))
    print(f"  ✓ Fit successful: R²={result.stats['r2']:.6f}, cost={result.stats['cost']:.4e}")
    for name in true_params:
        fitted = result.params.get(name, 0)
        true_val = true_params[name]
        err = abs(fitted - true_val) / abs(true_val)
        print(f"    {name}: true={true_val:.6e}, fitted={fitted:.6e}, error={err:.2%}")


if __name__ == "__main__":
    print("=" * 50)
    print("TLM V2.0 Core Tests")
    print("=" * 50)

    print("\n--- Parser ---")
    test_parser_single_component()
    test_parser_series()
    test_parser_parallel()
    test_parser_complex()
    test_collect_names()
    test_get_params()
    test_defaults()
    test_compute_simple()
    test_series_then_parallel()
    test_parse_error()

    print("\n--- Models ---")
    test_models_create()
    test_z_type_non_faradaic()
    test_e_type()
    test_universal_type()

    print("\n--- Fitting ---")
    test_fit_simple()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED ✓")
    print("=" * 50)
