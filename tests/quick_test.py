"""Quick test - no PyQt dependency"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.component_parser import parse, compute_z, get_parameter_names, SeriesNode, ParallelNode, ComponentNode
import numpy as np

def verify(name, ok):
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}")
    return ok

all_ok = True

# Test 1: Single component
n = parse('R1')
all_ok &= verify("R1 is ComponentNode", isinstance(n, ComponentNode) and n.name == 'R1')

# Test 2: Simple series
n = parse('R1-R2')
all_ok &= verify("R1-R2 is SeriesNode(R1,R2)",
    isinstance(n, SeriesNode) and len(n.children) == 2)

# Test 3: Simple parallel
n = parse('(R1R2)')
all_ok &= verify("(R1R2) is ParallelNode(R1,R2)",
    isinstance(n, ParallelNode) and len(n.children) == 2)

# Test 4: Complex
n = parse('R1-(R2Wo1)')
all_ok &= verify("R1-(R2Wo1) is Series(R1, Parallel(R2,Wo1))",
    isinstance(n, SeriesNode) and len(n.children) == 2 and
    isinstance(n.children[1], ParallelNode))

# Test 5: Nested parallel
n = parse('(R1(R2R3))')
all_ok &= verify("(R1(R2R3)) is Parallel(R1, Parallel(R2,R3))",
    isinstance(n, ParallelNode) and len(n.children) == 2 and
    isinstance(n.children[1], ParallelNode))

# Test 6: Faradaic template
n = parse('R3-Wo1')
all_ok &= verify("R3-Wo1 for zB faradaic",
    isinstance(n, SeriesNode) and len(n.children) == 2)

n = parse('(R2C2)')
all_ok &= verify("(R2C2) for zP flat electrode",
    isinstance(n, ParallelNode) and len(n.children) == 2)

# Test 7: Parameter extraction
n = parse('R1-(R2Wo1)-C1')
params = get_parameter_names(n)
for p in ['R1_R', 'R2_R', 'Wo1_Rd', 'Wo1_tau', 'C1_C', 'C1_p']:
    all_ok &= verify(f"  parameter {p}", p in params)

# Test 8: Impedance computation
n = parse('R1')
z = compute_z(n, 1000, {'R1_R': 100.0})
all_ok &= verify("R1 = 100 Ohm", abs(z - 100.0) < 1e-10)

n = parse('R1-R2')
z = compute_z(n, 1000, {'R1_R': 100.0, 'R2_R': 50.0})
all_ok &= verify("R1+R2 = 150 Ohm", abs(z - 150.0) < 1e-10)

n = parse('(R1R2)')
z = compute_z(n, 1000, {'R1_R': 100.0, 'R2_R': 100.0})
all_ok &= verify("R1//R2 = 50 Ohm", abs(z - 50.0) < 1e-10)

# Test 9: Series then parallel
n = parse('(R1-R2C1)')
z = compute_z(n, 1000, {'R1_R': 100.0, 'R2_R': 200.0, 'C1_C': 1e-6, 'C1_p': 1.0})
all_ok &= verify("(R1-R2C1) finite Z", np.isfinite(z) and abs(z) > 0)

# Test 10: Model creation  
from core.models import create_model
from core.models.base import MAIN_MODELS, BOUNDARY_CONDITIONS

for mt in MAIN_MODELS:
    for bc in BOUNDARY_CONDITIONS[mt]:
        model = create_model(mt, bc)
        all_ok &= verify(f"Create {mt}/{bc}", model is not None)

# Test 10: Z-type computation
model = create_model('Z-type', 'open-open', zC_zero=True)
model.set_component('zA', 'R1')
model.set_component('zB', 'C1')
f_arr = np.array([1000, 100, 10])
params = {'R1_R': 1.0, 'C1_C': 1e-3, 'C1_p': 0.8, 'R0': 0.5, 'L': 0.01}
Z = model.compute_Z(f_arr, params)
all_ok &= verify("Z-type open-open zC=0 produces finite Z",
    len(Z) == 3 and np.all(np.isfinite(Z)))

# Test 11: E-type computation
model = create_model('E-type', 'open-open')
model.set_component('zA', 'R1')
model.set_component('zB', 'C1')
model.set_component('zC', 'R2')
f_arr = np.array([1000, 100, 10])
params = {'R1_R': 1.0, 'R2_R': 1.0, 'C1_C': 1e-3, 'C1_p': 0.8, 'R0': 0.5, 'L': 0.01}
Z = model.compute_Z(f_arr, params)
all_ok &= verify("E-type open-open produces finite Z",
    len(Z) == 3 and np.all(np.isfinite(Z)))

# Test 12: Fitting
from core.fitting_engine import FittingEngine, FitParams

model = create_model('Z-type', 'open-open', zC_zero=True)
model.set_component('zA', 'R1')
model.set_component('zB', 'C1')

f_arr = np.logspace(3, -1, 20)
true_params = {'R1_R': 5.0, 'C1_C': 2e-3, 'C1_p': 0.85, 'R0': 1.0, 'L': 0.005}
z_true = model.compute_Z(f_arr, true_params)
z_noisy = z_true + 0.01 * np.random.randn(len(z_true)) + 0.01j * np.random.randn(len(z_true))

fit_p = FitParams(
    names=['R1_R', 'C1_C', 'C1_p', 'R0', 'L'],
    initial=[1.0, 1e-3, 0.5, 0.1, 0.001],
    lower=[1e-3, 1e-9, 0.1, 1e-4, 1e-6],
    upper=[100, 1.0, 1.0, 10.0, 1.0],
    fixed=[False, False, False, False, False],
)
engine = FittingEngine(model, f_arr, z_noisy)
result = engine.run(fit_p)
all_ok &= verify(f"Fit R2={result.stats['r2']:.6f}",
                 result.success and result.stats['r2'] > 0.8)

print(f"\n{'='*40}")
print(f"{'ALL TESTS PASSED [OK]' if all_ok else 'SOME TESTS FAILED [FAIL]'}")
print(f"{'='*40}")
