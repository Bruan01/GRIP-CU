[CmdletBinding()]
param(
    [switch]$DryRun,
    [string]$PythonExecutable = 'python'
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $ProjectRoot
$env:PYTHONPATH = $ProjectRoot
$env:PYTHONUTF8 = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
New-Item -ItemType Directory -Force -Path artifacts\logs | Out-Null

$logFile = Join-Path $ProjectRoot ("artifacts\logs\02_train_lora_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
$args = @(
  'scripts/train_nell23k_lora.py',
  '--task-file', 'artifacts/nell23k_paper/tasks.json',
  '--adapter-dir', 'artifacts/nell23k_paper/lora_adapter',
  '--trainer-output-dir', 'artifacts/nell23k_paper/trainer',
  '--seed', '2026', '--model-name', 'qwen-7b', '--model-source', 'local', '--model-cache-dir', 'model_cache', '--local-files-only',
  '--lora-r', '4', '--lora-alpha', '8', '--target-modules', 'down_proj', 'up_proj', 'gate_proj',
  '--num-train-epochs', '1', '--involve-qa-epochs', '10',
  '--s1-stop-loss-threshold', '0.15', '--s2-stop-loss-threshold', '0.15', '--s1-min-epoch', '1', '--s2-min-epoch', '1',
  '--per-device-train-batch-size', '1', '--gradient-accumulation-steps', '512', '--learning-rate', '1e-3',
  '--weight-decay', '1e-4', '--adam-beta1', '0.9', '--adam-beta2', '0.98', '--adam-epsilon', '1e-8', '--max-grad-norm', '1.0'
)
Write-Host "Stage 2 only: train and save LoRA. Log: $logFile"
if ($DryRun) { Write-Host "$PythonExecutable $($args -join ' ')"; exit 0 }
& $PythonExecutable @args 2>&1 | Tee-Object -FilePath $logFile
if ($LASTEXITCODE -ne 0) { throw "Stage 2 failed with exit code $LASTEXITCODE. Log: $logFile" }
