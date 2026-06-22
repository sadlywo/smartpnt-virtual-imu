# IMU Figure Style

## Sizes

- Single column: `3.5 x 2.4 in`
- Double column: `7.2 x 3.6 in`
- Tall diagnostic: `7.2 x 5.0 in`

## Typography

- Font family: DejaVu Sans
- Axis label: 8-9 pt
- Tick label: 7-8 pt
- Legend: 7-8 pt
- Title: avoid unless the figure is a standalone report graphic

## Lines and Colors

- High reference: dark neutral
- Low-grade IMU: muted red/orange
- Virtual IMU: blue/green
- Use line width 1.0-1.5 for time series and 1.5-2.0 for summary curves.
- Avoid transparent visual effects that export poorly to PDF.

## Required Labels

- Gyro: `Angular rate [rad/s]`
- Accel: `Specific force [m/s^2]`
- Time: `Time [s]`
- Frequency: `Frequency [Hz]`
- PSD gyro: `PSD [(rad/s)^2/Hz]`
- PSD accel: `PSD [(m/s^2)^2/Hz]`
- Allan gyro: `Allan deviation [rad/s]`
- Allan accel: `Allan deviation [m/s^2]`

## Captions

Caption drafts should state: signal source, comparison target, metric or plot type, and the key interpretation. Do not overclaim when high-grade IMU is only a proxy reference.
