import pytest

def test_openenv_yaml_valid():
    """Verify openenv.yaml is parseable and has required fields."""
    import yaml
    with open("openenv.yaml") as f:
        data = yaml.safe_load(f)
    assert data["spec_version"] == 1
    assert data["name"] == "codered_env"
    assert data["app"] == "server.app:app"
    assert data["port"] == 8000

def test_environment_subclass():
    """Verify CodeRedEnvironment is a valid OpenEnv Environment."""
    from openenv.core.env_server.interfaces import Environment
    from codered_env.server.codered_environment import CodeRedEnvironment
    assert issubclass(CodeRedEnvironment, Environment)

def test_all_action_types_importable():
    """All action types can be imported."""
    from codered_env.server.models.actions import (
        DispatchAmbulance, PrepareOR, PageSpecialist, AssignHospital,
        PreemptOR, AllocateBlood, TransferBlood, RequestMutualAid,
        QueryBloodType, QueryORStatus, MaintainPlan,
    )
    # Smoke test instantiation
    DispatchAmbulance(ambulance_id="AMB_1", target_node="NH45_BYPASS")
    MaintainPlan()
    RequestMutualAid()
