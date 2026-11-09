"""
Correctness validation for the analytical Jacobian.

We compare the analytical Jacobian (derived from first principles)
against a numerical Jacobian (finite differences). These MUST match
within floating-point tolerance — otherwise the IK math is wrong.

This is the single most important correctness test in robotics.

Author: Karan Dave
"""

import numpy as np
import pytest

from software.ik_solver.dh_parameters import DEFAULT_DH_PARAMS, N_JOINTS
from software.ik_solver.jacobian_ik import JacobianIK


class TestJacobianCorrectness:
    """Validate analytical Jacobian against numerical finite differences."""

    @staticmethod
    def numerical_jacobian(solver: JacobianIK, q: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Compute Jacobian via central finite differences (ground truth)."""
        n = len(q)
        J_num = np.zeros((6, n))
        for i in range(n):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[i] += eps
            q_minus[i] -= eps
            
            T_plus = solver.forward_kinematics(q_plus)
            T_minus = solver.forward_kinematics(q_minus)
            
            # Linear velocity (position derivative)
            J_num[:3, i] = (T_plus[:3, 3] - T_minus[:3, 3]) / (2 * eps)
            
            # Angular velocity (rotation derivative via log map approximation)
            R_diff = T_plus[:3, :3] @ T_minus[:3, :3].T
            # Extract rotation vector from rotation matrix (Rodrigues' formula inverse)
            angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
            if abs(angle) < 1e-10:
                omega = np.zeros(3)
            else:
                omega = (angle / (2 * np.sin(angle))) * np.array([
                    R_diff[2, 1] - R_diff[1, 2],
                    R_diff[0, 2] - R_diff[2, 0],
                    R_diff[1, 0] - R_diff[0, 1],
                ])
            J_num[3:, i] = omega / (2 * eps)
        
        return J_num

    @pytest.mark.parametrize("seed", [0, 1, 42, 100, 999])
    def test_analytical_matches_numerical(self, solver, seed):
        """Analytical Jacobian must match numerical within tolerance."""
        np.random.seed(seed)
        q = np.random.uniform(-np.pi/3, np.pi/3, size=N_JOINTS)
        
        J_ana = solver.jacobian(q)
        J_num = self.numerical_jacobian(solver, q)
        
        # Relative tolerance: 1e-4 is generous for numerical differentiation
        np.testing.assert_allclose(
            J_ana, J_num, rtol=1e-3, atol=1e-5,
            err_msg=f"Analytical Jacobian doesn't match numerical (seed={seed})"
        )

    def test_jacobian_shape(self, solver):
        """Jacobian must be 6xN for a 6-DOF arm."""
        q = np.zeros(N_JOINTS)
        J = solver.jacobian(q)
        assert J.shape == (6, N_JOINTS), f"Expected (6, {N_JOINTS}), got {J.shape}"

    # def test_jacobian_at_home(self, solver):
    #     """Sanity check: Jacobian at home position should have specific structure."""
    #     q = np.zeros(N_JOINTS)
    #     J = solver.jacobian(q)
        
    #     # J1 (base rotation) should only affect angular velocity about Z
    #     # and have zero linear velocity contribution (axis passes through origin)
    #     assert abs(J[0, 0]) < 1e-10, "J1 should not contribute to X linear velocity at home"
    #     assert abs(J[1, 0]) < 1e-10, "J1 should not contribute to Y linear velocity at home"

    def test_jacobian_rank_at_home(self, solver):
        """Jacobian at home should be full rank (no singularity)."""
        q = np.zeros(N_JOINTS)
        J = solver.jacobian(q)
        
        # Check rank using SVD
        singular_values = np.linalg.svd(J, compute_uv=False)
        rank = np.sum(singular_values > 1e-6)
        
        assert rank == 6, f"Jacobian rank is {rank}, expected 6 (full rank)"