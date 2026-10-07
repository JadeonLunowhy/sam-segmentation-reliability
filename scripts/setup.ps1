$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectPath
New-Item -ItemType Directory -Force -Path 'tmp' | Out-Null
$env:TEMP = Join-Path $projectPath 'tmp'
$env:TMP = $env:TEMP
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
}
& '.venv/Scripts/python.exe' -m ensurepip --upgrade
if ($LASTEXITCODE -ne 0) { throw 'pip setup failed' }
if (Test-Path -LiteralPath 'requirements-lock.txt') {
    & '.venv/Scripts/python.exe' -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
} else {
    & '.venv/Scripts/python.exe' -m pip install numpy pillow scipy matplotlib scikit-learn reportlab imageio-ffmpeg pyarrow
    if ($LASTEXITCODE -ne 0) { throw 'Scientific dependency installation failed' }
    & '.venv/Scripts/python.exe' -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'GPU dependency installation failed' }
}
if (-not (Test-Path -LiteralPath 'third_party/segment-anything/segment_anything')) {
    git clone https://github.com/facebookresearch/segment-anything.git third_party/segment-anything
    if ($LASTEXITCODE -ne 0) { throw 'SAM source download failed' }
    git -C third_party/segment-anything checkout dca509fe793f601edb92606367a655c15ac00fdf
    if ($LASTEXITCODE -ne 0) { throw 'SAM source pin failed' }
}
Write-Output 'Environment ready. Run the data download/preparation commands in README.md.'
