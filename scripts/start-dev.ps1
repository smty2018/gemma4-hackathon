<#
.SYNOPSIS
    Start the full local dev stack against the GPU-backed RunPod pod.

.DESCRIPTION
    The Gemma model can only run on the pod's GPU (this laptop has no CUDA device),
    so the FastAPI backend must run on the pod. This script:
      1. Ensures uvicorn is running on the pod (starts it detached if not).
      2. Opens an SSH tunnel so http://127.0.0.1:8000 on this machine reaches the
         pod's FastAPI server. Streamlit's API_BASE_URL already defaults to that.
      3. Optionally launches the Streamlit app locally.

    Re-run this any time after a reboot or a dropped terminal.

.PARAMETER SshHost
    The SSH host alias from ~/.ssh/config. Defaults to "runpod".

.PARAMETER NoStreamlit
    Skip launching Streamlit (only ensure the backend + tunnel are up).

.EXAMPLE
    ./scripts/start-dev.ps1
#>
param(
    [string]$SshHost = "runpod",
    [int]$Port = 8000,
    [switch]$NoStreamlit
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$RemoteRoot = "/workspace/gemma4-hackathon"

function Write-Step($message) {
    Write-Host "==> $message" -ForegroundColor Cyan
}

# 1. Ensure uvicorn is running on the pod ------------------------------------
Write-Step "Checking FastAPI backend on '$SshHost'..."
$remoteCheck = ssh $SshHost "pgrep -f 'uvicorn app.main:app' > /dev/null && echo running || echo stopped"
if ($remoteCheck -match "running") {
    Write-Host "    Backend already running on the pod."
} else {
    Write-Step "Starting uvicorn on the pod (detached)..."
    $startCmd = @"
cd $RemoteRoot/services/api && \
set -a && source $RemoteRoot/.env && set +a && \
source .venv/bin/activate && \
nohup uvicorn app.main:app --host 0.0.0.0 --port $Port > /workspace/uvicorn.log 2>&1 < /dev/null &
"@
    ssh $SshHost $startCmd | Out-Null
    Start-Sleep -Seconds 3
}

# Wait for the pod-side health endpoint to answer.
Write-Step "Waiting for backend health check on the pod..."
$healthy = $false
foreach ($attempt in 1..30) {
    $code = ssh $SshHost "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$Port/api/v1/health"
    if ($code -eq "200") { $healthy = $true; break }
    Start-Sleep -Seconds 2
}
if (-not $healthy) {
    throw "Backend did not become healthy on the pod. Check: ssh $SshHost 'tail -40 /workspace/uvicorn.log'"
}
Write-Host "    Backend healthy on the pod." -ForegroundColor Green

# 2. (Re)open the SSH tunnel -------------------------------------------------
Write-Step "Opening SSH tunnel 127.0.0.1:$Port -> pod:$Port ..."
# Close any tunnel we previously started on this port.
Get-CimInstance Win32_Process -Filter "Name='ssh.exe'" |
    Where-Object { $_.CommandLine -match "-L ${Port}:" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

Start-Process ssh -ArgumentList @("-N", "-L", "$Port`:127.0.0.1:$Port", $SshHost) -WindowStyle Hidden
Start-Sleep -Seconds 3

$tunnel = Test-NetConnection -ComputerName 127.0.0.1 -Port $Port
if (-not $tunnel.TcpTestSucceeded) {
    throw "Tunnel did not come up on port $Port."
}
Write-Host "    Tunnel is up: http://127.0.0.1:$Port reaches the pod." -ForegroundColor Green

# 3. Launch Streamlit --------------------------------------------------------
if ($NoStreamlit) {
    Write-Step "Skipping Streamlit (--NoStreamlit). Backend + tunnel are ready."
    return
}

Write-Step "Launching Streamlit..."
$streamlitDir = Join-Path $RepoRoot "apps/streamlit"
$streamlitPython = Join-Path $streamlitDir ".venv/Scripts/python.exe"
if (-not (Test-Path $streamlitPython)) {
    Write-Warning "Streamlit venv not found at $streamlitPython."
    Write-Warning "Backend + tunnel are ready; start Streamlit yourself when the venv exists."
    return
}

Push-Location $streamlitDir
try {
    & $streamlitPython -m streamlit run app.py
} finally {
    Pop-Location
}
