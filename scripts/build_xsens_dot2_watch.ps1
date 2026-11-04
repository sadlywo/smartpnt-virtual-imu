[CmdletBinding()]
param(
    [string]$FreeCADCmd = 'E:\FreeCAD\bin\FreeCADCmd.exe',
    [string]$Config = 'configs\cad\xsens_dot2_watch.json',
    [string]$OutputRoot = 'outputs\cad\xsens_dot2_watch'
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$freecad = (Resolve-Path -LiteralPath $FreeCADCmd).Path
$configPath = (Resolve-Path -LiteralPath (Join-Path $repo $Config)).Path
$packageRoot = (Resolve-Path -LiteralPath (Join-Path $repo 'cad\freecad')).Path
$macroPath = (
    Resolve-Path -LiteralPath (Join-Path $repo 'cad\freecad\Build_Xsens_DOT2_Watch.FCMacro')
).Path
$outputPath = Join-Path $repo $OutputRoot
$priorRuns = @()
if (Test-Path -LiteralPath $outputPath) {
    $priorRuns = @(
        Get-ChildItem -LiteralPath $outputPath -Directory | Select-Object -ExpandProperty FullName
    )
}

Write-Host "READ freecad=$freecad"
Write-Host "READ config=$configPath"
Write-Host "READ package_root=$packageRoot"
Write-Host "READ macro=$macroPath"
Write-Host "WRITE output_root=$outputPath"
Write-Host "SKIP prior_run_count=$($priorRuns.Count) reason=existing builds are immutable"

$previousConfig = $env:XSENS_DOT2_CONFIG
$previousOutputRoot = $env:XSENS_DOT2_OUTPUT_ROOT
$previousPackageRoot = $env:XSENS_DOT2_PACKAGE_ROOT
$previousMacro = $env:XSENS_DOT2_MACRO
$env:XSENS_DOT2_CONFIG = $configPath
$env:XSENS_DOT2_OUTPUT_ROOT = $outputPath
$env:XSENS_DOT2_PACKAGE_ROOT = $packageRoot
$env:XSENS_DOT2_MACRO = $macroPath
try {
    Push-Location (Split-Path -Parent $freecad)
    try {
        $bootstrap = "import os, runpy, sys; sys.path.insert(0, os.environ['XSENS_DOT2_PACKAGE_ROOT']); runpy.run_path(os.environ['XSENS_DOT2_MACRO'], run_name='__main__')"
        & $freecad -c $bootstrap
        if ($LASTEXITCODE -ne 0) {
            throw "FreeCAD generator failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
} finally {
    if ($null -eq $previousConfig) {
        Remove-Item Env:XSENS_DOT2_CONFIG -ErrorAction SilentlyContinue
    } else {
        $env:XSENS_DOT2_CONFIG = $previousConfig
    }
    if ($null -eq $previousOutputRoot) {
        Remove-Item Env:XSENS_DOT2_OUTPUT_ROOT -ErrorAction SilentlyContinue
    } else {
        $env:XSENS_DOT2_OUTPUT_ROOT = $previousOutputRoot
    }
    if ($null -eq $previousPackageRoot) {
        Remove-Item Env:XSENS_DOT2_PACKAGE_ROOT -ErrorAction SilentlyContinue
    } else {
        $env:XSENS_DOT2_PACKAGE_ROOT = $previousPackageRoot
    }
    if ($null -eq $previousMacro) {
        Remove-Item Env:XSENS_DOT2_MACRO -ErrorAction SilentlyContinue
    } else {
        $env:XSENS_DOT2_MACRO = $previousMacro
    }
}

$latest = Get-ChildItem -LiteralPath $outputPath -Directory |
    Where-Object { $_.FullName -notin $priorRuns } |
    Sort-Object Name -Descending |
    Select-Object -First 1
if (-not $latest) {
    throw 'Generator completed without a new timestamped output directory'
}
Write-Host "WRITE run_dir=$($latest.FullName)"
uv run python (Join-Path $repo 'cad\freecad\tests\verify_build.py') $latest.FullName
if ($LASTEXITCODE -ne 0) {
    throw "Artifact verification failed with exit code $LASTEXITCODE"
}
