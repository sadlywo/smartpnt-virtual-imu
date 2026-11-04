from __future__ import annotations

import csv
import math
from pathlib import Path

import FreeCAD as App
import MeshPart
import Part

from .assembly import Component
from .config import WatchParameters
from .validation import validate_step, validate_stl


def _feature(doc, name, shape):
    obj = doc.addObject("Part::Feature", name)
    obj.Shape = shape.copy()
    return obj


def export_build(
    doc,
    components: list[Component],
    output_dir: Path,
    params: WatchParameters,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    step_dir = output_dir / "step" / "components"
    petg_dir = output_dir / "stl" / "petg"
    tpu_dir = output_dir / "stl" / "tpu"
    for path in (step_dir, petg_dir, tpu_dir):
        path.mkdir(parents=True, exist_ok=True)

    fcstd = output_dir / "xsens_dot2_watch.FCStd"
    doc.recompute()
    doc.saveAs(str(fcstd))

    component_steps = []
    stl_files = []
    csv_rows = []
    for component in components:
        if component.name == "Polymer_Pin_PETG_2":
            shared_step = step_dir / "Polymer_Pin_PETG.step"
            csv_rows.append(
                [
                    component.name,
                    str(shared_step.relative_to(output_dir)).replace("\\", "/"),
                    "Polymer_Pin_PETG.SLDPRT",
                    *component.translation_mm,
                    str(component.fit_check),
                    str(component.print_config),
                ]
            )
            continue
        part_doc = App.newDocument(f"Export_{component.name}")
        obj = _feature(part_doc, component.name, component.shape)
        part_doc.recompute()
        step_path = step_dir / f"{component.name}.step"
        Part.export([obj], str(step_path))
        validate_step(step_path, component.shape)
        component_steps.append(str(step_path))
        if component.fit_check or component.name == "DOT2_Fit_Dummy":
            csv_rows.append(
                [
                    component.name,
                    str(step_path.relative_to(output_dir)).replace("\\", "/"),
                    f"{component.name}.SLDPRT",
                    *component.translation_mm,
                    str(component.fit_check),
                    str(component.print_config),
                ]
            )
        App.closeDocument(part_doc.Name)
        if component.printable:
            material_dir = tpu_dir if component.material == params.tpu_material else petg_dir
            stl_path = material_dir / f"{component.name}.stl"
            mesh = MeshPart.meshFromShape(
                Shape=component.shape,
                LinearDeflection=params.stl_linear_deflection_mm,
                AngularDeflection=math.radians(params.stl_angular_deflection_deg),
                Relative=False,
            )
            mesh.write(str(stl_path))
            validate_stl(stl_path, component.shape)
            stl_files.append(str(stl_path))

    assembly_objects = []
    for obj in doc.Objects:
        if (
            obj.TypeId == "Part::Feature"
            and not obj.Shape.isNull()
            and obj.Name != "DOT2_Fit_Coupon_PETG"
        ):
            assembly_objects.append(obj)
    assembly_step = output_dir / "step" / "Xsens_DOT2_Watch_Assembly.step"
    Part.export(assembly_objects, str(assembly_step))
    validate_step(assembly_step)

    import_csv = output_dir / "solidworks_import_manifest.csv"
    with import_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "component_name",
                "step_path",
                "part_filename",
                "x_mm",
                "y_mm",
                "z_mm",
                "include_fit_check",
                "include_print",
            ]
        )
        writer.writerows(csv_rows)
    return {
        "fcstd": str(fcstd),
        "assembly_step": str(assembly_step),
        "import_csv": str(import_csv),
        "component_steps": component_steps,
        "stl_files": stl_files,
    }
