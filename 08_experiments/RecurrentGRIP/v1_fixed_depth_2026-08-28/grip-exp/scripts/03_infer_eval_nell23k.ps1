[CmdletBinding()]
param(
    [switch]$Resume,
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
$logFile = Join-Path $ProjectRoot ("artifacts\logs\03_infer_eval_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
$args = @(
  'scripts/infer_eval_nell23k.py',
  '--input-file', 'outputs/data/nell23k/processed_test.json',
  '--adapter-dir', 'artifacts/nell23k_paper/lora_adapter',
  '--output-file', 'outputs/grip_inf/nell23k/nell23k_qwen_paper_3090.jsonl',
  '--seed', '2026', '--model-name', 'qwen-7b', '--model-source', 'local', '--model-cache-dir', 'model_cache', '--local-files-only',
  '--tokenize-max-length', '4096', '--gen-max-length', '1000'
)
if ($Resume) { $args += '--resume' }
Write-Host "Stage 3 only: infer and evaluate. Log: $logFile"
if ($DryRun) { Write-Host "$PythonExecutable $($args -join ' ')"; exit 0 }
& $PythonExecutable @args 2>&1 | Tee-Object -FilePath $logFile
if ($LASTEXITCODE -ne 0) { throw "Stage 3 failed with exit code $LASTEXITCODE. Rerun with -Resume to continue a partial JSONL prediction file." }
