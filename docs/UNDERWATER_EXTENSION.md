# Underwater Extension

## Why Virtual High-Grade IMU Matters Underwater

AUVs and ROVs often operate with limited GNSS access, intermittent acoustic positioning, strong vibration, and temperature/pressure variations. A virtual high-grade IMU can reduce short-horizon drift, improve dead reckoning during aiding outages, and provide uncertainty-aware inertial signals without the cost, size, and power demands of navigation-grade hardware.

## Error Distribution Differences

Compared with land vehicles, underwater platforms typically show stronger low-frequency motion, thruster-induced vibration, pressure-dependent environmental effects, and longer periods without global position updates. Magnetic sensing can be unreliable near steel structures, vehicles, or seabed anomalies. Temperature changes may be slower but more persistent, and vibration spectra are often tied to propeller RPM and control inputs.

## Recommended Extra Inputs

- Pressure/depth
- DVL velocity
- Thruster PWM/current
- Temperature
- Mission phase label

## Underwater Simulation Degradation Models

- Thruster vibration coupled to RPM/PWM
- Buoyancy-induced low-frequency motion
- Wave/current disturbance
- DVL partial or full outage
- Magnetic disturbance and unreliable magnetometer periods

## Transfer Strategy

1. Source pretraining on SmartPNT-MSF or paired terrestrial IMU data.
2. Synthetic underwater domain randomization for vibration, currents, DVL dropout, and thermal drift.
3. Few-shot fine-tuning on AUV/ROV paired or proxy-reference data.
4. Uncertainty-aware deployment with alarms when the virtual IMU is outside the trained domain.

## Public Dataset Links

- AUV navigation dataset: <https://github.com/nature1949/AUV_navigation_dataset>
- NTNU underwater visual-inertial dataset: <https://github.com/ntnu-arl/underwater-datasets>
- BeamsNet: <https://github.com/ansfl/BeamsNet>
- MissBeamNet: <https://github.com/ansfl/MissBeamNet>
