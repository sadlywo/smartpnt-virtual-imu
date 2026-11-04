from __future__ import annotations

import subprocess
import traceback
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import FreeCAD as App

from .artifacts import BuildLog, create_run_directory, sha256_file, write_json
from .assembly import build_components, create_assembly_document
from .config import load_parameters
from .exporter import export_build
from .validation import validate_components, validate_sensor_fit


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(environ: Mapping[str, str]) -> int:
    config_value = environ.get("XSENS_DOT2_CONFIG")
    output_value = environ.get("XSENS_DOT2_OUTPUT_ROOT")
    if not config_value or not output_value:
        print("FAIL environment reason=XSENS_DOT2_CONFIG and XSENS_DOT2_OUTPUT_ROOT are required")
        return 1
    config_path = Path(config_value).resolve()
    output_root = Path(output_value).resolve()
    run_dir = create_run_directory(output_root, datetime.now())
    log = BuildLog(run_dir / "build_log.txt")
    manifest_path = run_dir / "manifest.json"
    repo = config_path.parents[2]
    generator_path = Path(__file__).resolve().parents[1] / "Build_Xsens_DOT2_Watch.FCMacro"
    manifest = {
        "status": "building",
        "git_commit": _git_commit(repo),
        "freecad_version": list(App.Version()),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "generator_sha256": sha256_file(generator_path),
        "artifacts": [],
    }
    try:
        log.read(str(config_path))
        params = load_parameters(config_path)
        parameters_path = run_dir / "parameters.json"
        write_json(parameters_path, params.to_dict())
        log.write(str(parameters_path))
        components = build_components(params)
        validate_components(components)
        validate_sensor_fit(components)
        log.skip("Polymer_Pin_PETG_2.stl", "second assembly pin reuses the first pin STL")
        log.skip("DOT2_Fit_Dummy.stl", "reference component is not printable")
        log.skip("DOT2_Fit_Coupon_PETG in assembly", "fit coupon is calibration-only")
        manifest["components"] = [
            {
                "name": item.name,
                "material": item.material,
                "translation_mm": list(item.translation_mm),
                "bounding_box_mm": [
                    item.shape.BoundBox.XLength,
                    item.shape.BoundBox.YLength,
                    item.shape.BoundBox.ZLength,
                ],
                "printable": item.printable,
                "fit_check": item.fit_check,
                "print_config": item.print_config,
            }
            for item in components
        ]
        doc = create_assembly_document(components)
        result = export_build(doc, components, run_dir, params)
        App.closeDocument(doc.Name)
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file() or path.name in {"build_log.txt", "manifest.json"}:
                continue
            manifest["artifacts"].append(
                {
                    "relative_path": path.relative_to(run_dir).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
            log.write(str(path))
        manifest["status"] = "complete"
        manifest["parameters"] = params.to_dict()
        manifest["exports"] = result
        write_json(manifest_path, manifest)
        log.write(str(manifest_path))
        print(f"BUILD_COMPLETE run_dir={run_dir}")
        return 0
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["traceback"] = traceback.format_exc()
        write_json(manifest_path, manifest)
        log.fail("build", manifest["error"])
        print(f"BUILD_FAILED run_dir={run_dir} error={manifest['error']}")
        return 1
