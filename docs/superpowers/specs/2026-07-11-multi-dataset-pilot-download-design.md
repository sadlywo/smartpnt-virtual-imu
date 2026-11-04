# Multi-Dataset Pilot Download Design

## Goal

Add a small, reproducible pilot corpus of synchronized heterogeneous IMU data without downloading complete multi-modal datasets. The pilot must preserve source files, expose usable IMU data, and record enough provenance to reproduce or audit every download.

## Scope

The pilot contains:

- NavINST: one outdoor vehicle trajectory, restricted to bags containing the KVH-1750 reference IMU and available commercial-grade IMUs such as Xsens, Livox, or ZED IMUs.
- PennCOSYVIO: Tango Top consumer IMU data and the VI Sensor ADIS16488 reference IMU data.
- FusionPortable: the `20220209_Static_IMUs_3h20mins` sequence and one handheld dynamic sequence. If the source cannot be downloaded by sensor topic, retain the compressed rosbag and extract only IMU topics for the usable view.

Complete camera, LiDAR, radar, and full-dataset downloads are out of scope.

## Storage Layout

Each dataset uses the same raw-data layout:

```text
data/raw/<dataset>_pilot/
|-- 00_archives/       # Original downloaded files, unchanged
|-- 01_extracted/      # Extracted or topic-filtered IMU files
|-- _manifests/        # Source records, checksums, inventory, and notes
`-- README.md           # Local usage and provenance summary
```

The concrete roots are:

- `data/raw/navinst_pilot`
- `data/raw/penncosyvio_pilot`
- `data/raw/fusionportable_pilot`

All raw data remains ignored by Git under the existing `data/raw/*` rule.

## Download Configuration

A YAML configuration under `configs/` defines each artifact:

- dataset and sequence identifiers;
- source page and direct download URL;
- destination archive path;
- expected or observed size;
- optional SHA256 checksum;
- extraction mode;
- sensor names and nominal rates;
- synchronization and license notes.

The downloader reads this configuration and logs every read, write, verified skip, retry, and unsupported extraction step.

## Download Behavior

Downloads use resumable HTTP transfers where the host supports byte ranges. Existing files are skipped only after size or checksum validation. A partial or invalid file remains identifiable and is never silently treated as complete.

For hosts requiring browser-backed tokens or cloud-drive confirmation, the downloader first attempts the stable direct URL. If automated access is blocked, it writes an explicit blocked artifact record and continues with the other datasets instead of fabricating success.

## Extraction And Organization

Original archives are never modified. Standard ZIP, TAR, and 7z files are extracted into dataset-specific subdirectories under `01_extracted`.

PennCOSYVIO text files are preserved with their upstream sequence names. ROS bags remain intact unless an installed ROS-compatible tool can safely extract or filter IMU topics. Topic filtering is optional for the first pilot; the original bag and a manifest are sufficient when the local environment lacks the required ROS version.

## Manifest Contract

Each dataset receives a JSON manifest containing:

- source URL and source page;
- UTC download timestamp;
- local relative path;
- byte size and SHA256;
- download and extraction status;
- sequence, platform, low sensor, and reference sensor;
- nominal sampling rates;
- synchronization method;
- license or usage restriction notes;
- inventory of extracted files.

A top-level pilot summary records successful, skipped, and blocked artifacts and total disk usage.

## Failure Handling

- Network interruption: retain partial files and resume on the next run.
- URL expiration or authentication: mark the item blocked and continue.
- Checksum mismatch: keep the invalid file with an explicit status and do not extract it.
- Missing extraction tool: retain the original archive, report the missing dependency, and continue.
- Disk-space guard: refuse to start an artifact if estimated required space would reduce free space below 50 GB.

No existing files are deleted, renamed, or overwritten without validation.

## Verification

The pilot is complete when:

1. Every configured artifact has a manifest status of downloaded, verified-skip, or explicitly blocked.
2. Every downloaded artifact has a SHA256 value and nonzero size.
3. Extracted directories contain an inventory and preserve upstream sequence naming.
4. The top-level summary reports total bytes and per-dataset status.
5. A repeated downloader run performs only verified skips or resumes incomplete artifacts.
6. Existing ALiMU, SmartPNT, processed data, runs, and checkpoints remain unchanged.

## Follow-Up Boundary

Canonical coordinate transformation, sub-sample time alignment, lever-arm compensation, anti-alias resampling, and conversion to training Parquet are deliberately deferred to a separate data-preparation phase after the pilot schemas and timestamps have been inspected.
