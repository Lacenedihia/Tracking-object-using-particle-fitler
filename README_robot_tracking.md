# Robot Localization with a Particle Filter

Estimating the **position and orientation** of a mobile robot with a **particle filter** implemented in Python, despite uncertain movements and noisy sensors.

## Problem

A mobile robot does not know exactly where it is: its motion is imprecise and its sensor readings are noisy. The goal is to maintain a belief about its pose `(x, y, θ)` over time.

## Approach

The particle filter represents that belief with a cloud of particles, each one a possible pose of the robot. At every step:

1. **Predict**: move each particle with the robot's motion model, adding noise to represent movement uncertainty
2. **Update**: weight each particle by how well the sensor measurements fit its pose
3. **Resample**: keep the particles that explain the measurements best and discard the others
4. **Estimate**: the robot's pose is computed from the particle cloud

Over time the cloud concentrates around the true pose.

## Run it

```bash
pip install numpy matplotlib jupyter
jupyter notebook
```

Open the notebook and run the cells in order.

## Related project

The same idea applied to images: [Tracking objects in video with a particle filter](https://github.com/Lacenedihia/Tracking-object-using-particle-fitler).

## Author

**Dihia Lacene**: [GitHub](https://github.com/Lacenedihia) · [Portfolio](https://lacenedihia.netlify.app/)
