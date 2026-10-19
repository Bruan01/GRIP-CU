import argparse
import os
import os.path as osp
import subprocess
from typing import Optional

import torch
try:
    import wandb
except ImportError:
    wandb = None

from evaluation import auto_eval_batch
from models import get_inf_model
from utils import save_list_json, load_list_json

MP_INPUT_DIR = 'outputs/mp/input'
MP_OUTPUT_DIR = 'outputs/mp/output'

DATA_DIR = "outputs/data"
GRIP_INF_OUTPUT_DIR = "outputs/grip_inf"
LLM_INF_OUTPUT_DIR = "outputs/llm_inf"

def get_local_cache_root(rank: int):
    base = os.environ.get("SLURM_TMPDIR", "/tmp")
    job = os.environ.get("SLURM_JOB_ID", "nojob")
    user = os.environ.get("USER", "nouser")
    return os.path.join(base, f"grip_cache_{user}_{job}", f"rank_{rank}")


def get_visible_gpus():
    """Get list of visible GPU IDs from CUDA_VISIBLE_DEVICES."""
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible_devices is None:
        # If not set, assume all GPUs (rare)
        return list(range(torch.cuda.device_count()))
    else:
        # CUDA_VISIBLE_DEVICES="3,5,7"
        return [int(x.strip()) for x in cuda_visible_devices.split(',') if x.strip() != '']


def prepare_split_filename(dir_name: str, dataset: str, file_path: str, num_test: int, num_process: int):
    return [os.path.join(dir_name, f'{dataset}_{os.path.basename(file_path)}_num_{num_test}_{i}_in_{num_process}')
            for i in range(num_process)]


def run_command(
        rank: int,
        visible_gpus: list,
        args: argparse.Namespace,
        input_file: str,
        output_file: str,
        ref_file: Optional[str] = None, ):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(visible_gpus[rank])
    env["PYTHONPATH"] = "."
    cache_root = get_local_cache_root(rank)
    os.makedirs(cache_root, exist_ok=True)

    env["TRITON_CACHE_DIR"] = os.path.join(cache_root, "triton")
    env["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache_root, "inductor")

    command = (['python', args.script]
               + ['--rank', str(rank)]
               + args.subprocess_args \
               + ['--input_file', input_file,
                  '--output_file', output_file,
                  '--ref_file', ref_file])
    print('command: ', command)
    return subprocess.Popen(command, env=env)


def do_evaluation(
        metrics: list,
        output_file: str,
        llm_as_judge_model: Optional[str] = None):
    if "llm" in metrics:
        eval_model = get_inf_model(
                    llm_as_judge_model,
                    tokenize_max_length=5000,
                    gen_max_length=200)
    else:
        eval_model = None
    results = load_list_json(output_file)
    targets = []
    preds = []
    preds_no_sample = []
    questions = []
    for result in results:
        targets.append(result["target"])
        preds.append(result["response"])
        questions.append(result["question"])
        if "response_no_sample" in result:
            preds_no_sample.append(result["response_no_sample"])
    eval_results = auto_eval_batch(
        preds=preds,
        targets=targets,
        metrics=metrics,
        questions=questions,
        eval_model=eval_model,
    )
    for metric in metrics:
        print(f"Metric name: {metric}, metric value: {eval_results[metric]}")

    if preds_no_sample:
        eval_results = auto_eval_batch(
            preds=preds_no_sample,
            targets=targets,
            metrics=metrics,
            questions=questions,
            eval_model=eval_model,
        )
        print("No sampling generation result:")
        for metric in metrics:
            print(f"Metric name: {metric}, metric value: {eval_results[metric]}")
    return eval_results


def collect_output(output_file, mp_output_files, num_process):
    mp_output_res = [[] for _ in range(num_process)]
    for i, mp_output_file in enumerate(mp_output_files):
        mp_output_res[i] = load_list_json(mp_output_file)
        print(f"Process {i} collected {len(mp_output_res[i])} samples...")
    output = []
    len_output_max = max([len(o) for o in mp_output_res])
    for i in range(len_output_max):
        for j in range(num_process):
            if i >= len(mp_output_res[j]):
                continue
            if isinstance(mp_output_res[j][i], dict):
                output.append(mp_output_res[j][i])
            else:
                for d in mp_output_res[j][i]:
                    output.append(d)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_list_json(output_file, output)


def main(args):
    # set up directories
    os.makedirs(MP_INPUT_DIR, exist_ok=True)
    os.makedirs(MP_OUTPUT_DIR, exist_ok=True)

    input_file = osp.join(DATA_DIR, args.dataset, "processed_test.json")
    if osp.exists(osp.join(DATA_DIR, args.dataset, "processed_train.json")):
        ref_file = osp.join(DATA_DIR, args.dataset, "processed_train.json")
    elif osp.exists(osp.join(DATA_DIR, args.dataset, "processed_val.json")):
        ref_file = osp.join(DATA_DIR, args.dataset, "processed_val.json")
    else:
        ref_file = None

    if "run_grip" in args.script:
        output_base_dir = GRIP_INF_OUTPUT_DIR
    else:
        output_base_dir = LLM_INF_OUTPUT_DIR

    output_file = osp.join(output_base_dir, args.dataset, args.output_file)
    data = load_list_json(input_file)

    if args.num_test:
        data = [data[i] for i in range(args.num_test)]
    else:
        args.num_test = len(data)
    # check gpu availability and adjust num_process.
    visible_gpus = get_visible_gpus()
    if args.num_process > len(visible_gpus):
        print(f"num_process is set to {len(visible_gpus)} as only {len(visible_gpus)} GPUs are available.")
        args.num_process = len(visible_gpus)

    # split input data
    mp_input_data = [[] for _ in range(args.num_process)]
    for i, d in enumerate(data):
        mp_input_data[i % args.num_process].append(d)
    mp_input_files = prepare_split_filename(MP_INPUT_DIR, args.dataset, input_file, args.num_test, args.num_process)
    print('mp_input_files: ', mp_input_files)

    for mp_input_file, mp_input_d in zip(mp_input_files, mp_input_data):
        save_list_json(mp_input_file, mp_input_d)

    # prepare output filename
    mp_output_files = prepare_split_filename(MP_OUTPUT_DIR, args.dataset, output_file, args.num_test, args.num_process)
    print('mp_output_files: ', mp_output_files)

    # run subprocess
    processes = []
    for rank in range(args.num_process):
        processes.append(run_command(rank, visible_gpus, args, mp_input_files[rank], mp_output_files[rank], ref_file))

    return_codes = [p.wait() for p in processes]
    failed = [rank for rank, code in enumerate(return_codes) if code != 0]
    if failed:
        raise RuntimeError(f"GRIP subprocesses failed for rank(s) {failed}; no partial evaluation was run.")

    # collect output
    collect_output(output_file, mp_output_files, args.num_process)

    if args.do_eval:
        eval_results = do_evaluation(args.metrics, output_file, args.llm_as_judge_model)

        if args.report_to_wandb:
            if wandb is None:
                raise ImportError("wandb is not installed. Install it or omit --report_to_wandb.")
            wandb_run = wandb.init(project=args.wandb_project_name, name=args.wandb_run_name)
            wandb_run.config.update(vars(args))
            wandb_run.log(eval_results)
        else:
            for k, v in eval_results.items():
                print(f"{k}: {v}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', type=str, required=True)
    parser.add_argument('--num_process', type=int, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--num_test', type=int, default=None)
    parser.add_argument('--do_eval', action='store_true', )
    parser.add_argument('--metrics', nargs="+", type=str, default=["em", "f1"])
    parser.add_argument("--llm_as_judge_model", type=str, default="qwen-32b")
    parser.add_argument('--report_to_wandb', action='store_true', )
    parser.add_argument('--wandb_project_name', type=str, default='graph_lora')
    parser.add_argument('--wandb_run_name', type=str, default='mp')
    parser.add_argument('--subprocess_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    main(args)
