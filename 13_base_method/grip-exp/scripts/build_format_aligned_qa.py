"""Stage 1b: 生成「评测格式对齐」的 10-way 选择题 QA，修复 E12 的格式错配。

背景（E12 诊断结论）
------------------
原始训练数据 qa_samples 中 **0 条** 含
"Selected from the following candidate answers"，而评测集 4943 题 **100%** 含。
模型因此从未学会 10-way 选择题行为，导致 EM 仅 0.61%。

本脚本从训练图（train.txt 构成的单人图）中采样三元组，按**评测 process.py 的
逐字格式**构造 10-way 选择题 QA，答案使用正确的 ``concept:rel``（冒号）命名空间。

三种问法 + 生成开关：
  eval   —— 与 process.py 完全一致（word node + 候选列表）  ← 修复核心
  train  —— 原始 context_qa 问法（无候选列表）               ← 兼容性对照
  plain  —— 有候选列表但无 word node 措辞                     ← 消融

产出任务 JSON，字段与 generate_nell23k_tasks.py 一致，可直接喂给 Stage-2 训练。

用法
----
python scripts/build_format_aligned_qa.py \
    --input-file outputs/data/nell23k/processed_test.json \
    --out-task-file artifacts/nell23k_fmtalign/tasks.json \
    --num-qa 12014 --mode eval
"""

import argparse
import json
import os
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# 与 process.py / gen_bridge.py 逐字一致
Q_FORMAT_EVAL = ("What is the relation between word node {src} and word node {tgt}? "
                 "Selected from the following candidate answers: {candidates}.")
Q_FORMAT_TRAIN = "what is the relation between {src} and {tgt}?"
Q_FORMAT_PLAIN = ("What is the relation between {src} and {tgt}? "
                  "Selected from the following candidate answers: {candidates}.")

TEMPLATE = ("Given the context graph titled {title}, please answer the following question: "
            "{question} Response in the following format:<answer>[answer]</answer>")
ANSWER_FMT = "<answer>{answer}</answer>"

SYSTEM_PROMPTS = [
    "You are a thorough assistant. Your role is to offer concise, reliable, and clear guidance "
    "in response to user prompts. Do not speculate.",
    "You are a resourceful assistant. Your task is to provide direct, concise, and accurate "
    "guidance to user questions. Do not make assumptions.",
    "You are an insightful assistant. Your role is to give clear, concise, and correct responses "
    "to user prompts. Do not guess.",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-file', default='outputs/data/nell23k/processed_test.json')
    p.add_argument('--base-task-file', default='',
                   help='可选：原始 tasks.json。提供时保留其 context_samples，只替换 qa_samples。')
    p.add_argument('--out-task-file', default='artifacts/nell23k_fmtalign/tasks.json')
    p.add_argument('--num-qa', type=int, default=12014)
    p.add_argument('--mode', choices=('eval', 'train', 'plain', 'mix'), default='eval')
    p.add_argument('--way', type=int, default=10)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--preview', type=int, default=3)
    return p.parse_args()


def build_candidates(rng, correct_rel, all_rels, way):
    """从全部关系中采样 way-1 个干扰项 + 1 个正确项，随机打散顺序。"""
    non_target = [r for r in all_rels if r != correct_rel]
    if len(non_target) >= way - 1:
        neg = rng.permutation(non_target)[:way - 1].tolist()
    else:
        neg = non_target + rng.choice(non_target, way - 1 - len(non_target),
                                      replace=True).tolist()
    cands = rng.permutation(neg + [correct_rel]).tolist()
    return "; ".join(cands)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    # 载入图（processed_test.json 内含完整训练图）
    with open(args.input_file, encoding='utf-8') as f:
        record = json.loads(f.readline())
    graph, title = record['graph'], record['title']
    edge_list = graph['edge_list']            # [[src_text, rel, tgt_text], ...]
    print(f'图载入: nodes={len(graph["node_list"])} edges={len(edge_list)}')

    all_rels = sorted({e[1] for e in edge_list})
    print(f'关系总数: {len(all_rels)}')

    # ---- 采样并生成 10-way QA ----
    num = min(args.num_qa, len(edge_list))
    idx = rng.choice(len(edge_list), size=num, replace=False)
    modes = ('eval', 'train', 'plain') if args.mode == 'mix' else (args.mode,)

    qa_samples = []
    for i in idx:
        src, rel, tgt = edge_list[i]
        mode = modes[i % len(modes)] if len(modes) > 1 else modes[0]
        if mode == 'eval':
            q = Q_FORMAT_EVAL.format(src=src, tgt=tgt,
                                     candidates=build_candidates(rng, rel, all_rels, args.way))
        elif mode == 'plain':
            q = Q_FORMAT_PLAIN.format(src=src, tgt=tgt,
                                      candidates=build_candidates(rng, rel, all_rels, args.way))
        else:
            q = Q_FORMAT_TRAIN.format(src=src, tgt=tgt)
        sample = (f"<|im_start|>system\n{random.choice(SYSTEM_PROMPTS)}<|im_end|>\n"
                  f"<|im_start|>user\n{TEMPLATE.format(title=title, question=q)}<|im_end|>\n"
                  f"<|im_start|>assistant\n{ANSWER_FMT.format(answer=rel)}<|im_end|>\n\n")
        qa_samples.append(sample)

    print(f'生成 {len(qa_samples)} 条 10-way QA (mode={args.mode})')

    # ---- 组装任务文件 ----
    if args.base_task_file and Path(args.base_task_file).is_file():
        with open(args.base_task_file, encoding='utf-8') as f:
            base = json.load(f)
        context_samples = base.get('context_samples', [])
        print(f'复用 base context_samples: {len(context_samples)}')
    else:
        context_samples = []
        print('警告：未提供 --base-task-file，context_samples 为空。'
              'Stage-1 记忆将无法进行，建议提供原 tasks.json。')

    payload = {
        'format_version': 2,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'seed': args.seed,
        'source': {
            'input_file': str(Path(args.input_file).resolve()),
            'title': title,
            'graph_edges': len(edge_list),
            'graph_nodes': len(graph['node_list']),
        },
        'generation_config': {
            'num_qa': len(qa_samples), 'mode': args.mode, 'way': args.way,
            'note': 'format-aligned 10-way MCQ for GRIP NELL23K (fixes E12 mismatch)',
        },
        'context_samples': context_samples,
        'qa_samples': qa_samples,
    }

    out = Path(args.out_task_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, out)

    print('\n=== Stage 1b 完成 ===')
    print(f'输出:            {out.resolve()}')
    print(f'context_samples: {len(context_samples)}')
    print(f'qa_samples:      {len(qa_samples)}')

    if args.preview:
        print('\n--- 预览 ---')
        for s in qa_samples[:args.preview]:
            print(s)
            print('-' * 78)


if __name__ == '__main__':
    main()