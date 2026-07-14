from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MACRO = ROOT / "cad" / "solidworks" / "Import_Xsens_DOT2_Watch.bas"


def test_solidworks_macro_has_required_api_contract() -> None:
    text = MACRO.read_text(encoding="utf-8")
    required = [
        "Sub main()",
        "GetCurrentMacroPathName",
        "GetImportFileData",
        "LoadFile4",
        "ImportDiagnosis",
        "Extension.SaveAs",
        "AddComponent5",
        "AddConfiguration3",
        "SetComponentState",
        "FIT_CHECK",
        "PRINT",
        "READ ",
        "WRITE ",
        "SKIP ",
        "FAIL ",
    ]
    for token in required:
        assert token in text
    assert "E:\\" not in text
    assert "C:\\" not in text
