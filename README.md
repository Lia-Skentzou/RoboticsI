# Semester Project – Kinematic Modeling and Motion Simulation of an Industrial Robot Arm  
School of Electrical and Computer Engineering, NTUA  
Course: Robotics I – Analysis, Control, and Laboratory | 9th Semester

## Summary

This project implements the geometric and differential modeling of an industrial robot arm using Python. The goal is to analyze the robot's kinematic behavior, simulate its movement between two target points, and visualize the complete trajectory in 3D space.

## Key Objectives

- Define link parameters using Denavit-Hartenberg (DH) convention
- Derive forward and inverse kinematics equations
- Compute the Jacobian matrix and identify singular configurations
- Plan smooth point-to-point motion with time-parameterized polynomials
- Visualize joint angles, velocities, and 3D movement of the manipulator

## Features

- Symbolic and numeric computation using `sympy` and `numpy`
- Polynomial trajectory generation with velocity/acceleration constraints
- Animated 3D plot of the robot's full cycle (A → B → A)
- Joint-space and task-space plots over time

## Technologies Used

- Python 3.x
- NumPy, SymPy
- Matplotlib (2D/3D & animations)
- Jupyter Notebook or `.py` execution environment
