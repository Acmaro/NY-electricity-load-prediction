# One-time setup on Windows: creates .venv, installs CUDA torch (if an NVIDIA GPU is present) and the package.
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1          # install
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Data    # install + download data
param([switch]$Data, [string]$Python = "py -3.11", [string]$Cuda = "cu124")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path .venv)) {
    Write-Host "creating .venv with $Python"
    Invoke-Expression "$Python -m venv .venv"
}
$py = ".venv\Scripts\python.exe"
& $py -m pip install -q --upgrade pip

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "NVIDIA GPU found -> installing torch ($Cuda)"
    & $py -m pip install -q torch --index-url "https://download.pytorch.org/whl/$Cuda"
} else {
    Write-Host "no NVIDIA GPU -> installing CPU torch"
    & $py -m pip install -q torch
}
& $py -m pip install -q -e ".[dev]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }

& $py -m nyload doctor
if ($Data) { & $py -m nyload data --years 2013-2023 }
Write-Host "`nready: .\run.ps1 train"
