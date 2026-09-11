# start_system.ps1 - Windows equivalent of start_system.sh
# Launches both the Job Worker (background) and Live Scorer (foreground).
# Usage: .\start_system.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "==============================================="
Write-Host "  Starting Multi-Camera SJF Pipeline"
Write-Host "==============================================="
Write-Host ""

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
    Write-Host "[OK] Virtual environment: venv"
} elseif (Test-Path "mcenv\Scripts\Activate.ps1") {
    & "mcenv\Scripts\Activate.ps1"
    Write-Host "[OK] Virtual environment: mcenv"
} else {
    Write-Host "[WARN] No venv found - using system Python."
}

# Check Redis config
if (Test-Path ".env") {
    $redisHost = (Get-Content ".env" | Where-Object { $_ -match "^REDIS_HOST=" }) -replace "^REDIS_HOST=",""
    if ($redisHost -eq "localhost") {
        Write-Host ""
        Write-Host "[WARN] REDIS_HOST is set to localhost in .env"
        Write-Host "       Both the camera and display are on this machine - fine for local use."
        Write-Host ""
    } else {
        Write-Host "[OK] Redis target: $redisHost"
    }
}

# Set PYTHONPATH so imports resolve from project root
$env:PYTHONPATH = $PSScriptRoot


# Start Live Scorer in this window (foreground)
Write-Host "[>>] Launching Multi-Camera Live Scorer..."
Write-Host "     Press Q in the camera window to stop."
Write-Host ""
try {
    python live_scorer.py
} finally {
    Write-Host ""
    Write-Host "[!!] Camera feeds terminated."
    Write-Host "==============================================="
    Write-Host "  System Shutdown Complete."
    Write-Host "==============================================="
}
