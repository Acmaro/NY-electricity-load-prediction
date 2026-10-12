# Windows entry point: .\run.ps1 <doctor|data|train|eval|runs|test> [args...]
#   .\run.ps1 data --years 2013-2023
#   .\run.ps1 train model=lstm train.epochs=50
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "no .venv yet -> run: powershell -ExecutionPolicy Bypass -File scripts\setup.ps1"
    exit 1
}
Push-Location $PSScriptRoot
try { & $py -m nyload @args; exit $LASTEXITCODE } finally { Pop-Location }
