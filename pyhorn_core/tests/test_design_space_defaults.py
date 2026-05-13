from pyhorn_core.config.design_space import (
    FOLD_STYLES,
    ML_LRC_RANGE,
    ML_PROFILE_TYPES,
    ML_VTC_RANGE,
    MOUTH_AREA_RANGE,
    N_SEGMENTS_RANGE,
    OPTIMIZER_LRC_RANGE,
    OPTIMIZER_VTC_RANGE,
    PATH_LENGTH_RANGE,
    THROAT_AREA_RANGE,
)
from pyhorn_core.solver.optimizer import OptimizationConfig
from pyhorn_ml.core.space import DesignSpace


def test_optimizer_config_uses_shared_default_bounds():
    config = OptimizationConfig()

    assert config.throat_area_range == THROAT_AREA_RANGE
    assert config.mouth_area_range == MOUTH_AREA_RANGE
    assert config.path_length_range == PATH_LENGTH_RANGE
    assert config.lrc_range == OPTIMIZER_LRC_RANGE
    assert config.vtc_range == OPTIMIZER_VTC_RANGE


def test_ml_design_space_uses_shared_default_bounds():
    space = DesignSpace()

    assert space.throat_area_m2 == THROAT_AREA_RANGE
    assert space.mouth_area_m2 == MOUTH_AREA_RANGE
    assert space.path_length_m == PATH_LENGTH_RANGE
    assert space.lrc_m == ML_LRC_RANGE
    assert space.vtc_m3 == ML_VTC_RANGE
    assert space.n_segments == N_SEGMENTS_RANGE
    assert tuple(space.profile_type) == ML_PROFILE_TYPES
    assert tuple(space.fold_style) == FOLD_STYLES
