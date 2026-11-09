"""
Jacobian-based Inverse Kinematics for a 6-DOF serial manipulator.

Method: Damped Least Squares (Levenberg-Marquardt)
    dq = J^T (J J^T + lambda^2 I)^-1 * e

- Analytical geometric Jacobian (no finite differencing)
- Orientation error via rotation-matrix / axis-angle formulation
- Damping for singularity robustness
- Joint-limit clamping + step-size limiting

Run from repo root:
    python -m software.ik_solver.jacobian_ik

Author: Karan Dave
"""

from __future__ import annotations

import time
import numpy as np

from software.ik_solver.dh_parameters import (
    DEFAULT_DH_PARAMS, JOINT_LIMITS, N_JOINTS, DHParameter,
)


def dh_transform(theta: float, dh: DHParameter) -> np.ndarray:
    """Standard DH homogeneous transform (4x4)."""
    th = theta + dh.theta_offset
    ct, st = np.cos(th), np.sin(th)
    ca, sa = np.cos(dh.alpha), np.sin(dh.alpha)
    return np.array([
        [ct, -st * ca,  st * sa, dh.a * ct],
        [st,  ct * ca, -ct * sa, dh.a * st],
        [0.0,      sa,       ca,      dh.d],
        [0.0,     0.0,      0.0,       1.0],
    ])


class JacobianIK:
    """Damped-Least-Squares IK solver for a 6-DOF revolute arm."""

    def __init__(
        self,
        dh_params: list[DHParameter] | None = None,
        damping: float = 0.02,              # ← Changed from 0.05 to 0.02
        max_iterations: int = 300,          # ← Changed from 200 to 300
        pos_tolerance: float = 1e-4,        # 0.1 mm
        rot_tolerance: float = 1e-3,        # ~0.057 deg
        max_step: float = 0.25,             # ← Changed from 0.15 to 0.25 rad
        position_only: bool = False,
    ):
        self.dh = dh_params if dh_params is not None else DEFAULT_DH_PARAMS
        self.n = len(self.dh)
        self.damping = damping
        self.max_iterations = max_iterations
        self.pos_tolerance = pos_tolerance
        self.rot_tolerance = rot_tolerance
        self.max_step = max_step
        self.position_only = position_only
        self.last_iterations = 0

    # ---------------- Forward Kinematics ----------------
    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Return base->tool homogeneous transform for joint vector q."""
        T = np.eye(4)
        for theta, dh in zip(q, self.dh):
            T = T @ dh_transform(theta, dh)
        return T

    def joint_origins(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return joint axis vectors z_i and origins p_i (base frame)."""
        T = np.eye(4)
        origins = [T[:3, 3].copy()]
        axes = [T[:3, 2].copy()]
        for theta, dh in zip(q, self.dh):
            T = T @ dh_transform(theta, dh)
            origins.append(T[:3, 3].copy())
            axes.append(T[:3, 2].copy())
        return np.array(axes), np.array(origins)

    # ---------------- Geometric Jacobian ----------------
    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Analytical 6xN geometric Jacobian.
        For revolute joint i:
            Jv_i = z_(i-1) x (p_e - p_(i-1))
            Jw_i = z_(i-1)
        """
        axes, origins = self.joint_origins(q)
        p_e = origins[-1]
        J = np.zeros((6, self.n))
        for i in range(self.n):
            z, p = axes[i], origins[i]
            J[:3, i] = np.cross(z, p_e - p)
            J[3:, i] = z
        return J

    # ---------------- Task-space error ----------------
    @staticmethod
    def orientation_error(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
        """
        Axis-angle orientation error (3-vector, base frame).
        e_o = 0.5 * (n x n_d + s x s_d + a x a_d)
        Well-defined everywhere, unlike Euler-angle subtraction.
        """
        n_c, s_c, a_c = R_current[:, 0], R_current[:, 1], R_current[:, 2]
        n_d, s_d, a_d = R_target[:, 0], R_target[:, 1], R_target[:, 2]
        return 0.5 * (np.cross(n_c, n_d) + np.cross(s_c, s_d) + np.cross(a_c, a_d))

    def pose_error(self, q: np.ndarray, T_target: np.ndarray) -> np.ndarray:
        T = self.forward_kinematics(q)
        e_pos = T_target[:3, 3] - T[:3, 3]
        if self.position_only:
            return e_pos
        e_rot = self.orientation_error(T[:3, :3], T_target[:3, :3])
        return np.concatenate([e_pos, e_rot])

    # ---------------- Solver ----------------
    def solve(
        self,
        T_target: np.ndarray,
        q_init: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool]:
        """Solve IK. Returns (joint_angles_rad, converged)."""
        q = np.zeros(self.n) if q_init is None else np.array(q_init, dtype=float)
        rows = 3 if self.position_only else 6
        I = np.eye(rows)

        for it in range(self.max_iterations):
            e = self.pose_error(q, T_target)
            pos_err = np.linalg.norm(e[:3])
            rot_err = np.linalg.norm(e[3:]) if not self.position_only else 0.0

            if pos_err < self.pos_tolerance and rot_err < self.rot_tolerance:
                self.last_iterations = it
                return q, True

            J = self.jacobian(q)
            if self.position_only:
                J = J[:3, :]

            # DLS: solve (J J^T + lambda^2 I) y = e ; dq = J^T y
            # np.linalg.solve is preferred over explicit matrix inversion.
            JJt = J @ J.T + (self.damping ** 2) * I
            y = np.linalg.solve(JJt, e)
            dq = J.T @ y

            dq = np.clip(dq, -self.max_step, self.max_step)
            q = np.clip(q + dq, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

        self.last_iterations = self.max_iterations
        return q, False


# ---------------- Demo / self-check ----------------
def _demo() -> None:
    solver = JacobianIK()

    print("=" * 62)
    print("6-DOF Jacobian IK — self check")
    print("=" * 62)

    # Home configuration sanity check
    q_home = np.zeros(N_JOINTS)
    T_home = solver.forward_kinematics(q_home)
    print(f"\nHome tool position (m): {T_home[:3, 3].round(4)}")

    # Build a REACHABLE target from a known configuration (guaranteed solvable)
    q_true = np.deg2rad([20.0, 35.0, -40.0, 10.0, 25.0, -15.0])
    T_target = solver.forward_kinematics(q_true)
    print(f"Target tool position (m): {T_target[:3, 3].round(4)}")

    print("\nSolving...")
    t0 = time.perf_counter()
    q_sol, ok = solver.solve(T_target, q_init=np.zeros(N_JOINTS))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    T_sol = solver.forward_kinematics(q_sol)
    pos_err_mm = np.linalg.norm(T_sol[:3, 3] - T_target[:3, 3]) * 1000.0
    rot_err_deg = np.degrees(
        np.linalg.norm(solver.orientation_error(T_sol[:3, :3], T_target[:3, :3]))
    )

    print(f"\nConverged        : {ok}")
    print(f"Iterations       : {solver.last_iterations}")
    print(f"Solve time       : {elapsed_ms:.2f} ms")
    print(f"Solution (deg)   : {np.degrees(q_sol).round(2)}")
    print(f"Ground truth(deg): {np.degrees(q_true).round(2)}")
    print(f"Position error   : {pos_err_mm:.4f} mm")
    print(f"Orientation error: {rot_err_deg:.4f} deg")
    print("\nNote: multiple joint solutions can reach the same pose —")
    print("      differing from ground truth is fine if errors are ~0.")


if __name__ == "__main__":
    _demo()