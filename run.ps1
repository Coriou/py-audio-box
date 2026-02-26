#!/usr/bin/env pwsh
# run.ps1 — Windows PowerShell launcher for the voice-tools toolbox
#
# Usage:
#   .\run.ps1 <app-name> [args...]
#   $env:TOOLBOX_VARIANT = "gpu"; .\run.ps1 <app-name> [args...]
#
# Examples:
#   .\run.ps1 voice-split --url "https://www.youtube.com/watch?v=XXXX" --clips 5 --length 30
#   .\run.ps1 voice-synth speak --voice my-voice --text "Hello"   # CPU
#   $env:TOOLBOX_VARIANT = "gpu"
#   .\run.ps1 voice-synth speak --voice my-voice --text "Hello"   # GPU
#
# TOOLBOX_VARIANT:
#   (unset / "cpu")  — use docker-compose.yml only  (CPU image, default)
#   "gpu"            — overlay docker-compose.gpu.yml (CUDA image + GPU passthrough)
#                      Requires:  make build-gpu  +  NVIDIA Container Toolkit
#                      See docker-compose.gpu.yml for full setup instructions.
#
# Note: Git Bash users on Windows can use the ./run bash script instead.
#       This script is for native PowerShell (pwsh / Windows PowerShell 5.1+).

param(
    [Parameter(Mandatory = $false, Position = 0)]
    [string]$App = "",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArgs = @()
)

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $App) {
    Write-Host "Usage: .\run.ps1 <app-name> [args...]"
    Write-Host ""
    Write-Host "  `$env:TOOLBOX_VARIANT = 'gpu'; .\run.ps1 <app-name> [args...]   (GPU mode)"
    Write-Host ""
    Write-Host "Available apps:"
    Get-ChildItem "$ScriptRoot\apps" -Directory | ForEach-Object {
        Write-Host "  $($_.Name)"
    }
    exit 1
}

$Script    = "apps/$App/$App.py"
$ScriptFull = Join-Path $ScriptRoot $Script

if (-not (Test-Path $ScriptFull)) {
    Write-Error "Error: no script found at $Script"
    exit 1
}

# ── compose file selection ─────────────────────────────────────────────────────
# TOOLBOX_VARIANT=gpu adds the GPU overlay; unset or "cpu" uses CPU-only defaults.
[string[]]$ComposeFiles = @("-f", "docker-compose.yml")
if ($env:TOOLBOX_VARIANT -eq "gpu") {
    $ComposeFiles += "-f", "docker-compose.gpu.yml"
}

# Build the argument list for docker compose
# Using the call operator (&) with splatting keeps spaces in args intact.
& docker compose @ComposeFiles run --rm pab python $Script @AppArgs
