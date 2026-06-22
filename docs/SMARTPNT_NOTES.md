# SmartPNT-MSF Notes

SmartPNT-MSF is handled conservatively:

- Inspect directory structure and file previews before parsing.
- Support CSV/TXT, YAML/JSON calibration files, ROS bag/db3 placeholders, and unknown binary files.
- Require config-driven column mapping for low/high IMU sources.
- Estimate low/high time delay by cross-correlation where possible.
- Produce an alignment quality report before windowing.

If raw data is missing, commands should print clear placement instructions instead of failing with a stack trace.
