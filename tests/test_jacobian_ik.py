"""
Comprehensive tests for the Jacobian IK solver.

These tests validate:
- FK/IK round-trip consistency
- Joint limit enforcement
- Convergence on reachable targets
- Correct failure on unreachable targets
- Sub-millimeter accuracy
- Real-time performance
- Singularity robustness

Run with:  pytest tests/test_jacobian_ik.py -v
Run with coverage:  pytest tests/ --cov=software.ik_solver --cov-report=term-missing

Author: Karan Dave
"""

import time
import numpy as np
import pytest

from software.ik_solver.dh_parameters import JOINT_LIMITS, N_JOINTS, MAX_REACH
from software.ik_solver.jacobian_ik import JacobianIK


# ============================================================
# 1. FK/IK ROUND-TRIP CONSISTENCY
# ============================================================
class TestFKIKRoundTrip:
    """
    Given a known joint configuration, compute FK to get a pose,
    then solve IK for that pose. The solver must return a configuration
    that reaches the SAME pose (not necessarily the same angles —
    kinematic redundancy means multiple solutions exist).
    """

    def test_home_position(self, solver):
        """Home (all zeros) should solve trivially."""
        q_home = np.zeros(N_JOINTS)
        T_home = solver.forward_kinematics(q_home)
        q_sol, converged = solver.solve(T_home, q_init=np.zeros(N_JOINTS))
        
        assert converged, "Solver failed to converge on home position"
        np.testing.assert_allclose(q_sol, np.zeros(N_JOINTS), atol=1e-3)

    @pytest.mark.parametrize("seed", range(10))
    def test_random_reachable_poses(self, solver, random_reachable_pose, seed):
        """Test 10 different random reachable configurations."""
        np.random.seed(seed)
        q_true, T_target = random_reachable_pose()
        
        q_sol, converged = solver.solve(T_target, q_init=np.zeros(N_JOINTS))
        
        assert converged, f"Failed to converge on seed={seed}"
        
        # Verify the solution reaches the target pose
        T_sol = solver.forward_kinematics(q_sol)
        pos_err_mm = np.linalg.norm(T_sol[:3, 3] - T_target[:3, 3]) * 1000
        rot_err_deg = np.degrees(
            np.linalg.norm(solver.orientation_error(T_sol[:3, :3], T_target[:3, :3]))
        )
        
        assert pos_err_mm < 0.5, f"Position error {pos_err_mm:.3f}mm exceeds 0.5mm"
        assert rot_err_deg < 0.1, f"Orientation error {rot_err_deg:.3f}deg exceeds 0.1deg"

    def test_solution_respects_joint_limits(self, solver, random_reachable_pose):
        """Solver must NEVER return angles outside mechanical limits."""
        np.random.seed(42)
        for _ in range(20):
            _, T_target = random_reachable_pose()
            q_sol, converged = solver.solve(T_target)
            
            if converged:
                assert np.all(q_sol >= JOINT_LIMITS[:, 0] - 1e-6), "Joint min limit violated"
                assert np.all(q_sol <= JOINT_LIMITS[:, 1] + 1e-6), "Joint max limit violated"


# ============================================================
# 2. ACCURACY BENCHMARK
# ============================================================
class TestAccuracy:
    """Verify sub-millimeter precision on known targets."""

    def test_submillimeter_accuracy(self, strict_solver, random_reachable_pose):
        """With tight tolerances, error should be < 0.05mm."""
        np.random.seed(123)
        _, T_target = random_reachable_pose()
        
        q_sol, converged = strict_solver.solve(T_target)
        assert converged
        
        T_sol = strict_solver.forward_kinematics(q_sol)
        pos_err_mm = np.linalg.norm(T_sol[:3, 3] - T_target[:3, 3]) * 1000
        
        assert pos_err_mm < 0.05, f"Position error {pos_err_mm:.4f}mm exceeds 0.05mm"


# ============================================================
# 3. PERFORMANCE BENCHMARK
# ============================================================
class TestPerformance:
    """Ensure solver is fast enough for real-time control loops."""

    def test_solve_time_under_50ms(self, solver, random_reachable_pose):
        """Each solve must complete in < 50ms (real-time threshold)."""
        np.random.seed(99)
        times = []
        for _ in range(50):
            _, T_target = random_reachable_pose()
            t0 = time.perf_counter()
            solver.solve(T_target)
            times.append((time.perf_counter() - t0) * 1000)
        
        p50 = np.percentile(times, 50)
        p95 = np.percentile(times, 95)
        p99 = np.percentile(times, 99)
        
        print(f"\nPerformance (50 runs):")
        print(f"  P50: {p50:.2f} ms")
        print(f"  P95: {p95:.2f} ms")
        print(f"  P99: {p99:.2f} ms")
        
        assert p95 < 50, f"P95 latency {p95:.2f}ms exceeds 50ms real-time budget"


# ============================================================
# 4. UNREACHABLE TARGET HANDLING
# ============================================================
class TestUnreachableTargets:
    """Solver must honestly report failure for impossible poses."""

    def test_target_beyond_max_reach(self, solver):
        """Target farther than arm can reach must not converge."""
        T_target = np.eye(4)
        T_target[:3, 3] = [MAX_REACH + 0.5, 0, 0]  # 50cm beyond max reach
        
        _, converged = solver.solve(T_target)
        assert not converged, "Solver should NOT converge on unreachable target"

    def test_target_far_outside_workspace(self, solver):
        """Completely impossible target should fail gracefully."""
        T_target = np.eye(4)
        T_target[:3, 3] = [5.0, 5.0, 5.0]  # 5 meters away
        
        _, converged = solver.solve(T_target)
        assert not converged


# ============================================================
# 5. SINGULARITY ROBUSTNESS
# ============================================================
class TestSingularityRobustness:
    """Verify the damping factor prevents solver blow-up near singularities."""

    def test_near_fully_stretched_arm(self, solver):
        """
        Near-singularity: arm almost fully extended.
        Jacobian becomes ill-conditioned; damping must keep solver stable.
        """
        # J2 and J3 near zero = arm stretched forward (singular-ish)
        q_near_singularity = np.array([0.0, 0.01, 0.01, 0.0, 0.0, 0.0])
        T_target = solver.forward_kinematics(q_near_singularity)
        
        # Should not raise an exception (e.g., LinAlgError from inversion)
        q_sol, converged = solver.solve(T_target)
        # Either converges or fails gracefully — both are acceptable
        # The key is NO unhandled exception


# ============================================================
# 6. POSITION-ONLY MODE
# ============================================================
class TestPositionOnlyMode:
    """Test the 3-DOF position-only solver (ignores orientation)."""

    def test_position_only_converges_more_easily(self, solver):
        """Position-only should converge on targets that full-pose can't."""
        # Use a target with arbitrary orientation that may be hard to match
        q_true = np.deg2rad([45, 30, -20, 90, -45, 60])
        T_target = solver.forward_kinematics(q_true)
        
        # Perturb orientation significantly
        T_target[:3, :3] = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ], dtype=float)
        
        pos_only_solver = JacobianIK(position_only=True)
        q_sol, converged = pos_only_solver.solve(T_target)
        
        # Position-only should still converge (orientation is ignored)
        assert converged, "Position-only solver should converge"
        
        # Verify position matches (orientation will differ)
        T_sol = pos_only_solver.forward_kinematics(q_sol)
        pos_err_mm = np.linalg.norm(T_sol[:3, 3] - T_target[:3, 3]) * 1000
        assert pos_err_mm < 0.5