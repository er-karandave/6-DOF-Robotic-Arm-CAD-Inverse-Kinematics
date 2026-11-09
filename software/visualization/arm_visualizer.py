"""
3D Real-Time Visualization of the 6-DOF Robotic Arm.
Animates the transition from home position to the IK-solved target pose.

Run from repo root:
    python -m software.visualization.arm_visualizer

Author: Karan Dave
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from software.ik_solver.dh_parameters import DEFAULT_DH_PARAMS, N_JOINTS
from software.ik_solver.jacobian_ik import JacobianIK


class ArmVisualizer:
    def __init__(self):
        self.ik = JacobianIK()
        self.dh = DEFAULT_DH_PARAMS
        
        # Setup figure
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('6-DOF Robotic Arm — IK Visualization')
        
        # Set fixed workspace bounds for consistent viewing
        self.ax.set_xlim([-0.5, 0.5])
        self.ax.set_ylim([-0.5, 0.5])
        self.ax.set_zlim([0.0, 0.6])
        
        # Plot elements
        self.links, = self.ax.plot([], [], [], 'o-', lw=4, markersize=8, color='#1f77b4')
        self.target_point, = self.ax.plot([], [], [], 'r*', markersize=12, label='Target Pose')
        self.ax.legend()

    def get_link_positions(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate 3D coordinates of all joints for plotting."""
        T = np.eye(4)
        x, y, z = [T[0, 3]], [T[1, 3]], [T[2, 3]]
        
        for theta, dh_param in zip(q, self.dh):
            ct, st = np.cos(theta + dh_param.theta_offset), np.sin(theta + dh_param.theta_offset)
            ca, sa = np.cos(dh_param.alpha), np.sin(dh_param.alpha)
            
            T = T @ np.array([
                [ct, -st * ca,  st * sa, dh_param.a * ct],
                [st,  ct * ca, -ct * sa, dh_param.a * st],
                [0.0,      sa,       ca,      dh_param.d],
                [0.0,     0.0,      0.0,       1.0],
            ])
            x.append(T[0, 3])
            y.append(T[1, 3])
            z.append(T[2, 3])
            
        return np.array(x), np.array(y), np.array(z)

    def interpolate(self, q_start: np.ndarray, q_end: np.ndarray, steps: int = 50) -> list[np.ndarray]:
        """Generate smooth intermediate joint configurations."""
        return [q_start + (q_end - q_start) * (i / steps) for i in range(steps + 1)]

    def animate(self, q_start: np.ndarray, q_end: np.ndarray):
        """Create and display the animation."""
        trajectory = self.interpolate(q_start, q_end, steps=60)
        
        # Get target end-effector position for the red star
        T_target = self.ik.forward_kinematics(q_end)
        self.target_point.set_data([T_target[0, 3]], [T_target[1, 3]])
        self.target_point.set_3d_properties([T_target[2, 3]])

        def update(frame):
            q = trajectory[frame]
            x, y, z = self.get_link_positions(q)
            self.links.set_data(x, y)
            self.links.set_3d_properties(z)
            self.ax.set_title(f'6-DOF Robotic Arm — Step {frame}/{len(trajectory)-1}')
            return self.links,

        ani = FuncAnimation(self.fig, update, frames=len(trajectory), interval=30, blit=False)
        plt.show()


def main():
    print("Initializing 3D Visualizer...")
    vis = ArmVisualizer()
    
    # Start at home
    q_home = np.zeros(N_JOINTS)
    
    # Target from our previous successful IK run
    q_target = np.deg2rad([20.0, -2.31, 39.94, -32.63, 25.0, -15.0])
    
    print("Starting animation. Close the window to exit.")
    vis.animate(q_home, q_target)


if __name__ == "__main__":
    main()