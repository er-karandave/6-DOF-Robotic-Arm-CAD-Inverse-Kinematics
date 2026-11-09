"""
Quintic Polynomial Trajectory Planner for smooth, jerk-free robotic motion.

A 5th-order polynomial is used to guarantee C2 continuity:
- Continuous position
- Continuous velocity
- Continuous acceleration (zero jerk at boundaries)

This prevents mechanical shock, gear stripping, and sensor vibration in physical robots.

Author: Karan Dave
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrajectoryPoint:
    """Represents the state of a joint at a specific time."""
    position: float
    velocity: float
    acceleration: float


class QuinticTrajectory:
    """
    Generates smooth joint trajectories between a start and end state.
    
    q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    """

    def __init__(
        self,
        q_start: float,
        q_end: float,
        duration: float,
        v_start: float = 0.0,
        v_end: float = 0.0,
        a_start: float = 0.0,
        a_end: float = 0.0,
    ):
        """
        Args:
            q_start: Initial joint position (rad)
            q_end: Target joint position (rad)
            duration: Time to reach target (seconds)
            v_start: Initial velocity (rad/s), default 0
            v_end: Target velocity (rad/s), default 0
            a_start: Initial acceleration (rad/s^2), default 0
            a_end: Target acceleration (rad/s^2), default 0
        """
        if duration <= 0:
            raise ValueError("Duration must be strictly positive.")

        self.duration = duration
        
        # Pre-calculate polynomial coefficients for efficiency
        self.a0 = q_start
        self.a1 = v_start
        self.a2 = a_start / 2.0

        # Solve for a3, a4, a5 using boundary conditions at t = duration
        # Matrix form: A * x = B
        T = duration
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T

        A = np.array([
            [T3,    T4,    T5],
            [3*T2,  4*T3,  5*T4],
            [6*T,   12*T2, 20*T3]
        ])

        B = np.array([
            q_end - (self.a0 + self.a1 * T + self.a2 * T2),
            v_end - (self.a1 + 2 * self.a2 * T),
            a_end - (2 * self.a2)
        ])

        # Solve the linear system
        self.a3, self.a4, self.a5 = np.linalg.solve(A, B)

    def evaluate(self, t: float) -> TrajectoryPoint:
        """
        Evaluate position, velocity, and acceleration at time t.
        Clamps t to [0, duration] to prevent extrapolation errors.
        """
        t = np.clip(t, 0.0, self.duration)
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t

        # Position: q(t)
        q = self.a0 + self.a1*t + self.a2*t2 + self.a3*t3 + self.a4*t4 + self.a5*t5
        
        # Velocity: q'(t)
        v = self.a1 + 2*self.a2*t + 3*self.a3*t2 + 4*self.a4*t3 + 5*self.a5*t4
        
        # Acceleration: q''(t)
        a = 2*self.a2 + 6*self.a3*t + 12*self.a4*t2 + 20*self.a5*t3

        return TrajectoryPoint(position=q, velocity=v, acceleration=a)


class MultiJointTrajectoryPlanner:
    """
    Coordinates quintic trajectories across all N joints of a robotic arm,
    ensuring they all start and finish at the exact same time (synchronized motion).
    """

    def __init__(self, q_start: np.ndarray, q_end: np.ndarray, duration: float):
        if len(q_start) != len(q_end):
            raise ValueError("Start and end joint arrays must have the same length.")
        
        self.n_joints = len(q_start)
        self.duration = duration
        
        # Create a quintic trajectory for each joint
        self.joint_trajectories = [
            QuinticTrajectory(
                q_start=q_start[i],
                q_end=q_end[i],
                duration=duration,
                v_start=0.0, v_end=0.0,  # Start/stop at rest
                a_start=0.0, a_end=0.0   # Zero jerk at boundaries
            )
            for i in range(self.n_joints)
        ]

    def evaluate(self, t: float) -> np.ndarray:
        """Return the joint positions at time t."""
        return np.array([traj.evaluate(t).position for traj in self.joint_trajectories])
    
    def evaluate_all(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return positions, velocities, and accelerations at time t."""
        states = [traj.evaluate(t) for traj in self.joint_trajectories]
        positions = np.array([s.position for s in states])
        velocities = np.array([s.velocity for s in states])
        accelerations = np.array([s.acceleration for s in states])
        return positions, velocities, accelerations