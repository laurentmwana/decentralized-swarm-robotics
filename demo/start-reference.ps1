$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectDirectory
$bundledPython = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$demoDependencies = Join-Path $projectDirectory '.codex-test-deps'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
if ((Test-Path -LiteralPath $bundledPython) -and (Test-Path -LiteralPath $demoDependencies)) {
    $env:PYTHONPATH = $demoDependencies + [System.IO.Path]::PathSeparator + $env:PYTHONPATH
    & $bundledPython -m demo.reference_server @args
} else {
    python -m demo.reference_server @args
}
