[CmdletBinding()]
param(
    [switch]$ResumeOutputs,
    [switch]$DryRun,
    [string]$PythonExecutable = 'python'
)

# Run the paper-aligned Qwen2.5-7B / NELL23K reproduction on one GPU.
# Safe to rerun after interruption: generated tasks are reused from task_cache_dir.
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $ProjectRoot

$env:PYTHONPATH = $ProjectRoot
$env:PYTHONUTF8 = '1'
$env:TOKENIZERS_PARALLELISM = 'false'

$modelDir = Join-Path $ProjectRoot 'model_cache\Qwen--Qwen2.5-7B-Instruct'
$dataFile = Join-Path $ProjectRoot 'outputs\data\nell23k\processed_test.json'
$taskCache = Join-Path $ProjectRoot 'artifacts\task_cache\nell23k_paper_qwen'
$logDir = Join-Path $ProjectRoot 'artifacts\logs'
$outputFile = Join-Path $ProjectRoot 'outputs\grip_inf\nell23k\nell23k_qwen_paper_3090.json'
New-Item -ItemType Directory -Force -Path $logDir, $taskCache | Out-Null

if (-not (Test-Path $dataFile)) {
    throw "Missing processed NELL23K data: $dataFile`nRun: python scripts/process_raw_data.py --datasets nell23k --output_dir outputs/data --seed 2026"
}
if (-not (Test-Path (Join-Path $modelDir 'config.json')) -or -not (Test-Path (Join-Path $modelDir 'model-00004-of-00004.safetensors'))) {
    throw "Missing complete Qwen2.5-7B ModelScope cache: $modelDir`nRun: python scripts/download_model.py --model_name qwen-7b --model_cache_dir model_cache"
}

$overwrite = if ($ResumeOutputs) { 'False' } else { 'True' }
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile = Join-Path $logDir "nell23k_full_$timestamp.log"

$args = @(
    'scripts/mp_wrapper.py',
    '--script', 'scripts/run_grip.py',
    '--num_process', '1',
    '--dataset', 'nell23k',
    '--output_file', 'nell23k_qwen_paper_3090.json',
    '--do_eval',
    '--metrics', 'em', 'f1', 'hit',
    '--subprocess_args',
    '--overwrite', $overwrite,
    '--model_name', 'qwen-7b',
    '--model_source', 'local',
    '--model_cache_dir', 'model_cache',
    '--local_files_only', 'True',
    '--task_generator_model_name', 'qwen-7b',
    '--task_generator_use_vllm', 'False',
    '--task_generator_batch_size', '4',
    '--task_cache_dir', 'artifacts/task_cache/nell23k_paper_qwen',
    '--num_context_qa', '8000',
    '--num_reason_qa', '2000',
    '--num_summarization', '6000',
    '--sample_node_attribute_task', 'False',
    '--repharse_context_qa', 'False',
    '--context_upsampling', 'False',
    '--task_gen_max_length', '1000',
    '--tokenize_max_length', '4096',
    '--gen_max_length', '1000',
    '--quantization', 'False',
    '--lora_r', '4',
    '--lora_alpha', '8',
    '--target_modules', 'down_proj', 'up_proj', 'gate_proj',
    '--gather_batches', 'False',
    '--num_train_epochs', '1',
    '--involve_qa_epochs', '10',
    '--s1_stop_loss_threshold', '0.15',
    '--s2_stop_loss_threshold', '0.15',
    '--s1_min_epoch', '1',
    '--s2_min_epoch', '1',
    '--continue_training', 'False',
    '--remove_unused_columns', 'True',
    '--report_to', 'none',
    '--overwrite_output_dir', 'True',
    '--per_device_train_batch_size', '1',
    '--gradient_accumulation_steps', '512',
    '--learning_rate', '1e-3',
    '--weight_decay', '1e-4',
    '--adam_beta1', '0.9',
    '--adam_beta2', '0.98',
    '--adam_epsilon', '1e-8',
    '--max_grad_norm', '1.0',
    '--log_level', 'info',
    '--logging_strategy', 'steps',
    '--logging_steps', '1',
    '--save_strategy', 'no',
    '--bf16', 'True',
    '--tf32', 'False',
    '--gradient_checkpointing', 'False',
    '--lr_scheduler_type', 'linear',
    '--no_graph_context', 'True',
    '--use_subgraph', 'False',
    '--index_format', 'False'
)

Write-Host "Project root: $ProjectRoot"
Write-Host "Task cache:   $taskCache"
Write-Host "Log file:     $logFile"
Write-Host "Output file:  $outputFile"
Write-Host "Output resume mode: $ResumeOutputs (overwrite=$overwrite)"
Write-Host "Progress display: task-generation stages show [current/total] prompts; Trainer and inference show their own tqdm bars."
Write-Host "`nCommand: $PythonExecutable $($args -join ' ')`n"

if ($DryRun) {
    Write-Host 'Dry run only; no model process was started.'
    exit 0
}

& $PythonExecutable @args 2>&1 | Tee-Object -FilePath $logFile
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    throw "NELL23K reproduction exited with code $exitCode. Re-run this script to reuse task-cache entries. Log: $logFile"
}

if (-not (Test-Path $outputFile)) {
    throw "Run completed without the expected prediction file: $outputFile"
}
Write-Host "`nCompleted successfully. Metrics are near the end of: $logFile"
Write-Host "Predictions: $outputFile"
