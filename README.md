# Monte Carlo Localization (MCL) Using Particle Filter

## Overview
This project implements **Monte Carlo Localization (MCL)** using a **Particle Filter** for a **differential-drive robot** equipped with a **2D laser scanner**. The objective is to estimate the robot's position within a known map by recursively updating a set of particles representing possible locations.

## Methodology
The implementation consists of three main steps:

### 1. Motion Update (Odometry Model)
- Implements the function `sample_motion_model_odometry()`.
- Uses the **odometry motion model** to update particles based on control inputs.
- Introduces **Gaussian noise** to account for real-world motion uncertainty.
- Computes new positions `(x, y, θ)` for each particle.
- Ensures **angle normalization** within `[0, 2π]`.

### 2. Compute Weights (Observation Model)
- Implements the function `compute_weights()`.
- Uses **laser scan measurements** and a **likelihood map** to update particle weights.
- Converts scan data to **map coordinates** using `ranges2cells()`.
- Assigns **higher weights** to particles that better match the observed environment.

### 3. Resampling (Importance Sampling)
- Implements the function `resample()`.
- Uses **Systematic Resampling** to ensure particles with higher weights survive.
- Eliminates low-weight particles and normalizes the new set.

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install numpy matplotlib
```

## Usage
1. Load the provided dataset (map and laser scan data).
2. Execute the MCL algorithm:
   - Motion update using `sample_motion_model_odometry()`.
   - Weight computation using `compute_weights()`.
   - Resampling using `resample()`.
3. Visualize the estimated trajectory.

## Expected Outcome
- The particle distribution should converge towards the robot’s true location over time.
- High-weight particles should align with actual obstacles detected by the laser scanner.

## Notes
- The motion model assumes noisy odometry data.
- The observation model relies on a **likelihood field** for weight computation.
- Resampling ensures survival of the best particles, reducing degeneracy.

## References
- Thrun, S., Burgard, W., & Fox, D. (2005). **Probabilistic Robotics**.


