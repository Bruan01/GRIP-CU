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

$logFile = Join-Path $ProjectRoot ("artifacts\logs\01_generate_tasks_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
$args = @(
  'scripts/generate_nell23k_tasks.py',
  '--input-file', 'outputs/data/nell23k/processed_test.json',
  '--ref-file', 'outputs/data/nell23k/processed_val.json',
  '--task-file', 'artifacts/nell23k_paper/tasks.json',
  '--task-cache-dir', 'artifacts/task_cache/nell23k_paper_qwen',
  '--seed', '2026',
  '--model-name', 'qwen-7b', '--model-source', 'local', '--model-cache-dir', 'model_cache', '--local-files-only',
  '--task-generator-model-name', 'qwen-7b', '--no-task-generator-use-vllm', '--task-generator-batch-size', '4',
  '--num-context-qa', '8000', '--num-reason-qa', '2000', '--num-summarization', '6000',
  '--no-sample-node-attribute-task', '--no-repharse-context-qa', '--no-context-upsampling', '--format-as-instruction',
  '--task-gen-max-length', '1000', '--tokenize-max-length', '4096', '--involve-qa-epochs', '10'
)
Write-Host "Stage 1 only: create/cached GRIP training tasks. Log: $logFile"
Write-Host "Resume cache: artifacts\task_cache\nell23k_paper_qwen"
if ($DryRun) { Write-Host "$PythonExecutable $($args -join ' ')"; exit 0 }
& $PythonExecutable @args 2>&1 | Tee-Object -FilePath $logFile
if ($LASTEXITCODE -ne 0) { throw "Stage 1 stopped with exit code $LASTEXITCODE. Rerun this script: cached generation will resume." }
