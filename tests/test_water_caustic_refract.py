import pytest
import graphics.water_caustic_refract as mod

CASES = [
    ("refract_matrix", (1,)),
    ("caustic_order", (1,)),
    ("async_fence", (1,)),
    ("foam_follow", (1,)),
    ("underwater_depth", (1,)),
    ("caustic_phase", (1,)),
    ("normal_update", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("reflection_lag", (1,)),
    ("metrics", (1,)),
    ("wave_scale", (1,)),
    ("camera_cut", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
