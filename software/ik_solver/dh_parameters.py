"""
Denavit-Hartenberg (DH) kinematic model for the 6-DOF robotic arm.

Convention: Standard (distal) DH.
    T_i = Rz(theta_i) * Tz(d_i) * Tx(a_i) * Rx(alpha_i)

All lengths in METRES, all angles in RADIANS.
Author: Karan Dave
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DHParameter:
    """Standard DH parameters for a single revolute joint."""
    a: float                  # link length      (m)
    alpha: float              # link twist       (rad)
    d: float                  # link offset      (m)
    theta_offset: float = 0.0 # joint zero offset(rad)


# ------------------------------------------------------------------
# Prototype geometry — update these to match your SolidWorks model
# ------------------------------------------------------------------
BASE_HEIGHT   = 0.080   # base -> shoulder
UPPER_ARM     = 0.250   # shoulder -> elbow
FOREARM       = 0.220   # elbow -> wrist
TOOL_LENGTH   = 0.080   # wrist -> tool tip

DEFAULT_DH_PARAMS = [
    DHParameter(a=0.0,       alpha= np.pi / 2, d=BASE_HEIGHT),  # J1 base yaw
    DHParameter(a=UPPER_ARM, alpha=0.0,        d=0.0),          # J2 shoulder pitch
    DHParameter(a=FOREARM,   alpha=0.0,        d=0.0),          # J3 elbow pitch
    DHParameter(a=0.0,       alpha= np.pi / 2, d=0.0),          # J4 wrist roll
    DHParameter(a=0.0,       alpha=-np.pi / 2, d=0.0),          # J5 wrist pitch
    DHParameter(a=0.0,       alpha=0.0,        d=TOOL_LENGTH),  # J6 tool roll
]

# Mechanical joint limits (rad) — enforced in software AND ESP32 firmware
JOINT_LIMITS = np.array([
    [-np.pi,      np.pi    ],   # J1
    [-np.pi / 2,  np.pi / 2],   # J2
    [-np.pi / 2,  np.pi / 2],   # J3
    [-np.pi,      np.pi    ],   # J4
    [-np.pi / 2,  np.pi / 2],   # J5
    [-np.pi,      np.pi    ],   # J6
])

N_JOINTS = len(DEFAULT_DH_PARAMS)
MAX_REACH = BASE_HEIGHT + UPPER_ARM + FOREARM + TOOL_LENGTH  # sanity envelope