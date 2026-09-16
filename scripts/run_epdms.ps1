<#
.SYNOPSIS
    Orchestration script for NuRec 300 EPDMS Safety Evaluation Pipeline.
.DESCRIPTION
    Runs Phase 0 Audit, followed by condition evaluation (with resume/checkpoint),
    and finally computes summary statistics and research reports.
#>

param(
    [string]$Config = "configs\epdms_300.json",
    [string]$Profile = "nurec_safety_proxy_v1",
    [double]$Horizon = 4.0,
    [switch]$AuditOnly,
    [switch]$Resume = $true,
    [Nullable[int]]$MaxClips = $null
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " NuRec 300 EPDMS / Safety Evaluation Pipeline" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Run Phase 0 Audit
Write-Host "`n[1/3] Running Phase 0 Environment & Data Contract Audit..." -ForegroundColor Yellow
python .\scripts\audit_epdms_inputs.py --config $Config
if ($LASTEXITCODE -ne 0) {
    Write-Error "Audit failed. Halting pipeline."
}

if ($AuditOnly) {
    Write-Host "`n[+] AuditOnly specified. Exiting successfully." -ForegroundColor Green
    exit 0
}

# 2. Run Evaluation
Write-Host "`n[2/3] Running Trajectory Condition Evaluation..." -ForegroundColor Yellow
$evalArgs = @(".\scripts\evaluate_epdms.py", "--config", $Config, "--profile", $Profile, "--horizon", $Horizon)
if ($Resume) { $evalArgs += "--resume" }
if ($MaxClips -ne $null) { $evalArgs += @("--max-clips", $MaxClips) }

python @evalArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "Evaluation failed. Halting pipeline."
}

# 3. Run Summarization
Write-Host "`n[3/3] Generating Aggregates & Research Summaries..." -ForegroundColor Yellow
python .\scripts\summarize_epdms.py --config $Config
if ($LASTEXITCODE -ne 0) {
    Write-Error "Summarization failed."
}

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host " Pipeline Execution Completed Successfully!" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
