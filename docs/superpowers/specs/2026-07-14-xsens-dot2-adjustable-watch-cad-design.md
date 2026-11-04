# Xsens DOT 2 Adjustable Watch CAD Design

## Goal

Create a reproducible FDM-ready wrist mount for one Xsens DOT 2 sensor. The mount uses a removable top-load sensor cradle, two flexible TPU straps, and a reusable wrist-bottom ratchet clasp covering a nominal adult wrist circumference of 140–210 mm.

The primary geometry generator is a FreeCAD 0.21.1 Python macro that can be executed and validated in the current workspace. A secondary SolidWorks 2022 VBA import macro converts the generated STEP components into native `.SLDPRT` files and a `.SLDASM` assembly.

## Scope

The deliverable includes:

- a parameterized FreeCAD macro;
- a master `.FCStd` document containing the positioned assembly components;
- separate STEP and STL exports for every printable component;
- an assembly STEP for CAD exchange;
- a SolidWorks 2022 VBA import-and-assemble macro;
- build parameters, a manifest, a build log, and usage documentation.

The first version is a functional FDM prototype. It does not claim production snap-cycle life, certified skin-contact safety, waterproofing, or a validated fit across every printer and TPU formulation.

## Confirmed Design Decisions

| Decision | Selected value |
|---|---|
| Manufacturing process | FDM 3D printing |
| Nominal wrist range | 140–210 mm |
| Sensor retention | Top-load snap cradle |
| Strap architecture | Two flexible straps with wrist-bottom ratchet clasp |
| Cradle and clasp material | PETG |
| Strap material | TPU 95A |
| CAD organization | Multi-component assembly |
| Primary generator | FreeCAD 0.21.1 Python macro |
| SolidWorks path | STEP import and assembly macro for SolidWorks 2022 |
| Magnetic/metal hardware | None |

Metal pins, springs, and magnetic clasps are excluded because the DOT 2 contains a magnetometer. Printed polymer pins or straight sections of 1.75 mm polymer filament are used instead.

## Sensor Interface

The official Xsens DOT 2 envelope is 36.3 × 30.4 × 10.8 mm. The top surface contains the status LED and central power button; the Micro-USB charging port is on the bottom. The cradle therefore leaves the top surface open and treats removal from the cradle as the required charging workflow.

The generator uses these initial fit parameters:

| Parameter | Default |
|---|---:|
| Sensor length | 36.3 mm |
| Sensor width | 30.4 mm |
| Sensor height | 10.8 mm |
| Cavity length | 36.8 mm |
| Cavity width | 30.9 mm |
| Initial XY clearance | 0.25 mm per side |
| Sensor corner radius approximation | 6.0 mm |
| Cradle base thickness | 3.0 mm |
| Cradle wall thickness | 2.4 mm |
| Snap engagement | 0.6 mm |

The official envelope dimensions are authoritative, but the corner-radius approximation and FDM fit clearance must remain top-level parameters. Before extended use, the user must measure the physical sensor and print the generated fit coupon. The full cradle is not treated as a calibration coupon.

## Mechanical Components

### 1. `DOT2_Cradle_PETG`

The default cradle has a 44.8 × 39.0 mm main body, a 3.0 mm base, two strap lugs, four corner supports, and two short side snap hooks. The sensor is pressed down from the top and removed by flexing the short hooks through finger-access reliefs. The open top preserves direct access to the button and visibility of the LED.

The support geometry keeps the sensor envelope above the wrist-facing base. No charging cable pass-through is included. This avoids a weak opening in the base and prevents the cable from being loaded while the device is worn.

### 2. `Strap_Long_TPU`

The long strap is 120 × 20 × 2.2 mm and contains the reusable ratchet tooth field. The default tooth pitch is 4.0 mm, tooth height is 1.2 mm, and 19 usable lock positions are generated. The free end has a tapered lead-in.

### 3. `Strap_Short_TPU`

The short strap is 40 × 20 × 2.2 mm and connects the opposite cradle lug to the ratchet housing. Both straps use rounded external edges.

### 4. `Ratchet_Housing_PETG`

The default wrist-bottom housing is 28 × 24 × 7 mm. It guides the long TPU strap and constrains lateral motion while leaving a finger-accessible release area. The sliding clearance defaults to 0.35 mm per side.

### 5. `Ratchet_Pawl_PETG`

The pawl is an independently printable and replaceable flexure. Its tooth direction permits tightening and resists pull-out. Pressing the exposed release pad rotates or flexes the tooth clear of the TPU strap.

### 6. `Polymer_Pin_PETG`

One pin model is generated and instantiated twice. The nominal diameter is parameterized around 1.75 mm so printed pins or straight filament segments can be evaluated without introducing metal next to the IMU.

### 7. `DOT2_Fit_Dummy`

The non-print assembly placeholder represents the configurable DOT 2 envelope and carries reference markings for the top button, LED, and bottom charging-port face. It is present in the fit-check assembly state and excluded from print exports.

### 8. `DOT2_Fit_Coupon_PETG`

The fit coupon reproduces two adjacent cradle supports, one snap hook, the configured cavity offset, and the configured corner radius. It is exported as a separate PETG STL and excluded from the watch assembly. Its purpose is to calibrate the sensor fit without printing the complete cradle.

## Wrist-Range Model

The nominal loop calculation uses a 44.8 mm cradle path, 120 mm long strap, 40 mm short strap, and 28 mm clasp path. With a ratchet overlap from 20.8 to 92.8 mm, the generated 19 positions span 140–212 mm in 4 mm increments. This encloses the confirmed 140–210 mm target range while retaining one extra-large discrete position.

## Assembly States

The FreeCAD master document represents the assembly with named `App::Part` containers and positioned `Part::Feature` solids. FreeCAD 0.21.1 assembly-workbench constraints are not required for generation or export.

Two logical states are recorded in the manifest:

- `FIT_CHECK`: cradle, straps, clasp, pins, pawl, and the DOT 2 placeholder are visible and positioned for interference inspection.
- `PRINT`: the sensor placeholder is excluded; each printable component is placed at its own export origin and exported separately by material.

The SolidWorks macro imports component STEP files, saves each imported component as `.SLDPRT`, inserts the components into `Xsens_DOT2_Watch.SLDASM`, applies the known generated placements, and creates matching `FIT_CHECK` and `PRINT` configurations.

## Generation Architecture

The primary source file is `cad/freecad/Build_Xsens_DOT2_Watch.FCMacro`. It is valid Python for execution either from the FreeCAD macro UI or through `E:\FreeCAD\bin\FreeCADCmd.exe`.

The macro is divided into bounded units:

1. parameter parsing and validation;
2. logging and output-directory creation;
3. reusable rounded-profile and export helpers;
4. one geometry builder per component;
5. master-document placement;
6. shape and interference validation;
7. FCStd, STEP, and STL export;
8. manifest and final summary writing.

The SolidWorks source is `cad/solidworks/Import_Xsens_DOT2_Watch.bas`. It does not recreate the feature history. It reads the generated STEP paths and placements, imports them with SolidWorks 2022, saves native part documents, builds the assembly, and logs every read, write, skip, or failure.

## Build Flow and Artifacts

The default output directory is:

`outputs/cad/xsens_dot2_watch/<timestamp>/`

The timestamped directory is immutable from the generator's perspective. A new invocation creates a new directory rather than overwriting a previous build.

Each build contains:

- `xsens_dot2_watch.FCStd`;
- `step/components/*.step`;
- `step/Xsens_DOT2_Watch_Assembly.step`;
- `stl/petg/*.stl`;
- `stl/tpu/*.stl`;
- `parameters.json`;
- `manifest.json`;
- `build_log.txt`;
- `solidworks_import_manifest.csv`.

The manifest records the generator file hash, git commit when available, FreeCAD version, effective parameters, materials, component bounding boxes, validation results, and every exported file hash. Generated CAD, STEP, STL, logs, and output directories remain uncommitted.

## Error Handling

- Invalid dimensions, negative clearances, an inverted wrist range, or an insufficient tooth count fail before geometry is created.
- Every component builder returns one named solid or raises an error with the component and stage name.
- Boolean failures, invalid shapes, zero volumes, unexpected solid counts, or export failures stop the build.
- A failed build retains its partial timestamped directory and marks `manifest.json` as `failed`; it never deletes a previous run.
- Existing files are never silently overwritten. The build logs all files read, written, and skipped.
- The SolidWorks macro stops on a missing STEP file, import failure, save failure, or assembly insertion failure and records the last successful component.

## Validation

The FreeCAD generator must perform and report:

- `Shape.isValid()` for every component;
- non-zero volume and exactly one printable solid per exported component;
- expected bounding-box dimensions within a small numeric tolerance;
- no unintended volume intersection between the seated sensor placeholder and cradle;
- clearance checks at the minimum, middle, and maximum ratchet positions;
- STEP re-import followed by shape-count, validity, and bounding-box checks;
- non-empty STL meshes, closed-solid checks where supported, and millimetre-scale bounding-box checks;
- separation of PETG and TPU exports.

Static tests outside FreeCAD cover parameter validation, deterministic filenames, timestamp isolation, logging categories, manifest consistency, and rejection of incomplete exports. The generated preview is inspected visually from top, side, bottom, and exploded views.

SolidWorks runtime validation cannot be completed in the current environment because SolidWorks 2022 is not installed. The README therefore requires the user to run the import macro, rebuild the assembly, inspect the two configurations, and return `solidworks_build_log.txt` if any API call fails.

## FDM Prototype Guidance

Initial slicing assumptions are a 0.4 mm nozzle and 0.20 mm layer height. Cradle, housing, and pawl print in PETG; straps print flat in TPU 95A. The cradle prints base-down. The pawl is oriented so layer lines do not split across its primary flexure root.

The initial prototype sequence is:

1. print the DOT 2 fit coupon;
2. adjust XY clearance and corner radius if required;
3. print and cycle-test the pawl with a short ratchet strip;
4. print the complete component set;
5. verify the 140 mm, 175 mm, and 210 mm circumference positions;
6. perform repeated insertion, pull, and release tests before wearing during motion capture.

## Acceptance Criteria

The implementation is complete when:

- one command runs successfully with FreeCAD 0.21.1 and creates the full timestamped artifact set;
- all generated printable components pass the automated geometry and export checks;
- the master FCStd opens with named, positioned components;
- the sensor placeholder can be seated without unintended solid intersection while the snap hooks retain its upper edge;
- the 19-position ratchet representation spans 140–212 mm in 4 mm increments and therefore encloses the 140–210 mm nominal range;
- PETG and TPU STL files are exported separately;
- the PETG fit coupon is generated and excluded from the assembly;
- the SolidWorks VBA module imports the generated component STEP files and contains clear run instructions;
- all reads, writes, skips, parameters, and validation outcomes are traceable through the build log and manifest.

## References

- Movella, [Xsens DOT User Manual, Revision I](https://www.movella.com/hubfs/Xsens%20DOT%20User%20Manual.pdf).
- FreeCAD, [Python scripting basics](https://reqrefusion.github.io/FreeCAD-Documentation-html/wiki/FreeCAD_Scripting_Basics.html).
- FreeCAD, [Part API](https://reqrefusion.github.io/FreeCAD-Documentation-html/wiki/Part_API.html).
