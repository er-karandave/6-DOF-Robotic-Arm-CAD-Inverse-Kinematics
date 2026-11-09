"""
Shared pytest fixtures for IK solver tests.

Fixtures are reusable setup objects. Instead of recreating the solver
in every test, we define it once here and inject it via function arguments.

Author: Karan Dave
"""

import pytest
import numpy as np

from software.ik_solver.dh_parameters import DEFAULT_DH_PARAMS, N_JOINTS, JOINT_LIMITS
from software.ik_solver.jacobian_ik import JacobianIK


@pytest.fixture
def solver():
    """Default IK solver instance."""
    return JacobianIK()


@pytest.fixture
def strict_solver():
    """Solver with tighter tolerances for precision tests."""
    return JacobianIK(
        damping=0.01,
        max_iterations=300,
        pos_tolerance=1e-5,   # 0.01 mm
        rot_tolerance=1e-4,   # ~0.006 deg
    )


@pytest.fixture
def random_reachable_pose(solver):
    """
    Factory fixture: generates a guaranteed-reachable target pose
    by running FK on a random (valid) joint configuration.
    
    This is the gold standard for IK testing — we KNOW a solution exists.
    """
    def _generate():
        # Sample random joint angles within mechanical limits
        q_true = np.random.uniform(
            JOINT_LIMITS[:, 0] * 0.9,   # 90% of range to avoid edge singularities
            JOINT_LIMITS[:, 1] * 0.9,
        )
        T_target = solver.forward_kinematics(q_true)
        return q_true, T_target
    return _generate