# 6-DOF-Robotic-Arm-CAD-Inverse-Kinematics
6-DOF robotic arm featuring custom Jacobian-based Inverse Kinematics. Bridges SolidWorks mechanical design with Raspberry Pi C++/Python control logic for precise trajectory tracking.
SolidWorks, Python, C++, Raspberry Pi, Jacobian Matrices, PWM Motor Control, 3D Printing.

I took some help of AI for documentation, Learning, Research and Brainstorming.



# 📁 Repository Structure
6-DOF-Robotic-Arm-IK/
├── README.md                          # Main documentation
├── .gitignore
├── requirements.txt                   # Python dependencies
├── docs/
│   ├── math/
│   │   └── jacobian_ik_explainer.md   # Mathematical foundation
│   ├── hardware/
│   │   ├── bom.md                     # Bill of Materials
│   │   └── wiring_diagram.md          # ESP32 wiring guide
│   └── media/                         # GIFs, images, videos
├── cad/
│   ├── 6dof_arm.step                  # SolidWorks export
│   ├── 6dof_arm.stl                   # 3D printable STL
│   └── assembly_drawings.pdf
├── software/
│   ├── ik_solver/
│   │   ├── __init__.py
│   │   ├── dh_parameters.py           # Denavit-Hartenberg params
│   │   ├── forward_kinematics.py      # FK for verification
│   │   ├── jacobian_ik.py             # Jacobian-based IK solver
│   │   └── trajectory_planner.py      # Smooth trajectory generation
│   ├── hardware_interface/
│   │   ├── esp32_bridge.py            # Serial bridge to ESP32
│   │   └── gripper_control.py
│   └── visualization/
│       ├── arm_visualizer.py          # Matplotlib 3D visualization
│       └── real_time_plotter.py
├── firmware/
│   └── esp32_servo_controller/
│       ├── platformio.ini
│       ├── src/
│       │   └── main.cpp               # ESP32 PWM servo control
│       └── include/
│           └── config.h
└── tests/
    ├── test_fk_ik_consistency.py
    └── test_trajectory_smoothness.py






# 🤖 6-DOF Robotic Arm: CAD Design & Jacobian-Based Inverse Kinematics

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![ESP32](https://img.shields.io/badge/ESP32-Firmware-green.svg)](https://www.espressif.com/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-orange.svg)](https://docs.ros.org/en/humble/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete end-to-end robotics project combining **mechanical CAD design**, **Jacobian-based inverse kinematics**, and **embedded hardware control**. The system bridges SolidWorks mechanical modeling with a Python/C++ IK solver deployed on an ESP32 microcontroller driving 6 servo actuators via PWM.

![Robot Arm Demo](docs/media/arm_demo.gif)

---

## 🎯 Project Overview

This project demonstrates **full-system robotics ownership** across three critical domains:

| Domain | Implementation |
|--------|----------------|
| **Mechanical** | 6-DOF arm designed in SolidWorks, optimized for FDM 3D printing with integrated actuator packaging |
| **Mathematical** | Custom Jacobian-based IK solver using Denavit-Hartenberg parameters and damped least-squares method |
| **Embedded** | ESP32 firmware handling real-time PWM servo control with serial command protocol |

---

## ✨ Key Features

- 🦾 **6 Degrees of Freedom** — Full spatial positioning + orientation of end-effector
- 🧮 **Jacobian-Based IK Solver** — Damped Least Squares (DLS) method with singularity handling
- 📐 **Denavit-Hartenberg Parameters** — Rigorous kinematic modeling
- 🎬 **Smooth Trajectory Planning** — Quintic polynomial trajectories for jerk-free motion
- 🔌 **ESP32 Hardware Bridge** — Real-time PWM control of 6 servos via serial protocol
- 📊 **3D Visualization** — Matplotlib-based real-time arm visualization
- ✅ **FK/IK Consistency Tests** — Automated verification of solver accuracy

---

## 🏗️ System Architecture

┌─────────────────────────────────────────────────────────────────────────────┐
│                        USER / APPLICATION LAYER                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────┐   │
│  │  Target Pose     │    │  Trajectory      │    │  3D Visualization    │   │
│  │  (x,y,z, r,p,y)  │───▶│  Planner         │───▶│  (Matplotlib/RViz2)  │   │
│  └──────────────────┘    └──────────────────┘    └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPUTATION LAYER (Python/C++)                        │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────┐   │
│  │  DH Parameters   │───▶│  Forward         │───▶│  Jacobian IK Solver  │   │
│  │  (Kinematic      │    │  Kinematics      │    │  (DLS Method)        │   │
│  │   Model)         │    │  (Verification)  │    │  λ = 0.05            │   │
│  └──────────────────┘    └──────────────────┘    └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │ Joint Angles (rad → deg)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      COMMUNICATION LAYER (Serial UART)                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Protocol: "J1:45.00,J2:-30.00,J3:60.00,J4:0.00,J5:45.00,J6:0.00\n" │   │
│  │  Baud: 115200 | Frame: ASCII | ACK: "ACK:J1:45.00,...\n"             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │ UART (USB-TTL)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EMBEDDED FIRMWARE LAYER (ESP32 / C++)                    │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────┐   │
│  │  Serial Parser   │───▶│  Joint Limit     │───▶│  LEDC PWM Generator  │   │
│  │  (Command Decode)│    │  Enforcement     │    │  (50Hz, 16-bit)      │   │
│  └──────────────────┘    └──────────────────┘    └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │ PWM Signals (50Hz, 1-2.5ms pulse)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HARDWARE LAYER (Physical)                             │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────┐   │
│  │  6× Servo Motors │    │  3D-Printed      │    │  Power Supply        │   │
│  │  (MG996R)        │    │  Structure       │    │  (5V 10A)            │   │
│  └──────────────────┘    └──────────────────┘    └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘



# Data Flow Diagram
User Input (Target Pose)
        │
        ▼
┌───────────────────┐
│  IK Solver        │ ─── Computes joint angles via DLS method
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Trajectory       │ ─── Interpolates smooth path between waypoints
│  Planner          │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Serial Bridge    │ ─── Packages angles into ASCII command
└───────────────────┘
        │ UART
        ▼
┌───────────────────┐
│  ESP32 Firmware   │ ─── Parses, validates, generates PWM
└───────────────────┘
        │ PWM (50Hz)
        ▼
┌───────────────────┐
│  Servo Motors     │ ─── Physical movement of arm joints
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  FK Verification  │ ─── Reads actual pose, compares to target
│  (Optional)       │     (Closed-loop validation)
└───────────────────┘



---

## 🧮 Mathematical Foundation

The IK solver uses the **Damped Least Squares (DLS)** method, also known as the Levenberg-Marquardt approach:
Δθ = J^T · (J · J^T + λ²I)^(-1) · Δx


Where:
- `J` = 6×6 Jacobian matrix (computed via DH parameters)
- `λ` = Damping factor (prevents singularity issues)
- `Δx` = End-effector pose error
- `Δθ` = Joint angle correction

The solver iteratively converges to a solution with typical accuracy of **< 0.5mm** positional error.

📖 [Full Mathematical Explanation →](docs/math/jacobian_ik_explainer.md)

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone [https://github.com/er-karandave/6-DOF-Robotic-Arm-IK.git](https://github.com/er-karandave/6-DOF-Robotic-Arm-CAD-Inverse-Kinematics.git)
cd 6-DOF-Robotic-Arm-IK



# 🎓 Learning Outcomes
This project demonstrates competency in:
✅ Robot Kinematics — DH parameters, Jacobians, FK/IK
✅ Mechanical Design — CAD, actuator packaging, 3D printing DFM
✅ Embedded Systems — ESP32, PWM, serial protocols
✅ Numerical Methods — Iterative solvers, singularity handling
✅ System Integration — Bridging software, math, and hardware


# 🗺️ Roadmap
Add ROS 2 control interface (ros2_control + MoveIt2)
Implement camera-based visual servoing
Add force/torque sensor feedback for impedance control
Migrate IK solver to C++ for real-time performance
Implement pick-and-place demo with AprilTag detection
