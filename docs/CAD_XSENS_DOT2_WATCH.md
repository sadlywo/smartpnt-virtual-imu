# Xsens DOT 2 Adjustable Watch CAD

This project generates a parameterized, FDM-printable wrist mount for the Xsens DOT 2.
The nominal wrist range is 140–212 mm in 19 discrete 4 mm ratchet positions. The sensor
cradle, ratchet housing, pawl, and two reusable pins are PETG; the two straps are TPU 95A.
No metal or magnetic hardware is required.

## Generate with FreeCAD 0.21.1

From the repository root, run:

```powershell
& .\scripts\build_xsens_dot2_watch.ps1
```

The command reads `configs/cad/xsens_dot2_watch.json` and writes a new timestamped directory
under `outputs/cad/xsens_dot2_watch/`. It never overwrites a previous build. A completed run
contains:

- `xsens_dot2_watch.FCStd`;
- eight unique component STEP files and one positioned assembly STEP;
- seven unique printable STL files;
- `solidworks_import_manifest.csv`;
- `parameters.json`, `manifest.json`, and `build_log.txt`.

The seven STL models are the cradle, fit coupon, long strap, short strap, ratchet housing,
ratchet pawl, and reusable pin. Print two copies of `Polymer_Pin_PETG.stl`. Do not print the
sensor dummy STEP.

## Print order and fit calibration

1. Print `stl/petg/DOT2_Fit_Coupon_PETG.stl` first.
2. Measure the physical DOT 2 and test the coupon. Adjust `sensor_corner_radius_mm` or
   `xy_clearance_per_side_mm` in the JSON configuration if required, then generate a new run.
3. Print `Ratchet_Pawl_PETG.stl` and a short test section of the long TPU strap. Cycle the
   release and verify that the tooth locks and releases without cracking.
4. Print the cradle, both straps, housing, pawl, and two copies of the polymer pin.
5. Assemble the clasp and verify the 140 mm, 176 mm, and 212 mm positions before dynamic use.

Do not force the DOT 2 into a coupon that is visibly too tight. FDM shrinkage and extrusion
width vary by printer, filament, and slicer profile; the checked-in 0.25 mm clearance per side
is a starting value, not a universal calibration.

## Starting FDM settings

The geometry assumes a 0.4 mm nozzle and 0.20 mm layer height. Use millimetres when importing
STL files into the slicer.

For PETG, start with four walls, five top/bottom layers, and 30% gyroid infill. Use 100% infill
for the pawl and pins. For TPU 95A, start with four walls, a low volumetric flow, and a print
speed around 25–35 mm/s. The 2.2 mm straps will be largely solid with four walls.

Recommended orientations:

- cradle and fit coupon: flat base on the build plate;
- long and short TPU straps: flat body on the plate, ratchet teeth facing upward;
- ratchet housing: stand on the `x = 0` end with a brim, so its tunnel and snap channels rise
  vertically without internal support;
- pawl: stand on the release-pad end with a brim;
- pins: print horizontally at 0.12–0.16 mm layer height, 100% infill, and print spares.

Inspect the slicer preview. Add localized support only where required by the selected printer;
do not fill the housing tunnel or the two short-strap snap channels with support that cannot be
removed.

## Mechanical assembly

1. Align each TPU hinge hole with a cradle lug and insert one PETG pin from the side.
2. Push the two flexible arms at the end of the short TPU strap into the matching side channels
   at the entrance of the PETG ratchet housing. Push until both widened tips snap into their
   pockets.
3. Flex the PETG pawl through the top release window. Its tooth must point into the housing
   tunnel and face the ramp teeth of the long strap.
4. Feed the long TPU strap through the housing. Pull to tighten; press the pawl release pad while
   withdrawing the strap to loosen.
5. Insert the DOT 2 from above under the two retention hooks. The top button and LED face remain
   exposed.

Remove the sensor from the cradle before charging. No waterproofing, skin-contact certification,
impact certification, or sports-safety certification is claimed. Inspect the straps, pins,
snap arms, and pawl before every dynamic test and replace any part showing whitening, layer
separation, or cracking.

## Import into SolidWorks 2022

Create a new SolidWorks VBA macro, import
`cad/solidworks/Import_Xsens_DOT2_Watch.bas`, run `main`, and select a completed FreeCAD build
directory. The macro imports the generated STEP components, saves native SLDPRT files, creates
`FIT_CHECK` and `PRINT` configurations, and saves `Xsens_DOT2_Watch.SLDASM` beneath the selected
build directory. Inspect `solidworks_build_log.txt` before making downstream changes.

SolidWorks 2022 is not available in the generator environment, so the VBA module has static API
contract coverage but still requires runtime validation on a machine with SolidWorks 2022.

## Design reference and traceability

The approved design rationale is recorded in
[`docs/superpowers/specs/2026-07-14-xsens-dot2-adjustable-watch-cad-design.md`](superpowers/specs/2026-07-14-xsens-dot2-adjustable-watch-cad-design.md).
Each timestamped build records the configuration hash, generator hash, git commit, component
dimensions, artifact hashes, FreeCAD version, and read/write/skip log. Generated output paths are
intentionally not used as permanent documentation links because every build is immutable and has
a unique timestamp.
