"""E12 诊断：验证「训练/评测格式错配」是否为 EM 崩塌的主因。

要裁决的假设
------------
E12 (0.5B baseline) 实测 EM=0.61%，低于 10-way 随机猜测的 ~10%。
需要区分两种解释：

  H1「图已进参数，但被输出侧挡住」
     —— 训练/评测表面形式差异（无候选列表 / 无 word node）+
        Stage-2 答案类型失衡（仅 22.1% 教"输出冒号关系"）导致取不出来。
  H2「0.5B 容量装不下 NELL23K」—— 图根本没进去。

注意：表面形式差异来自 GRIP **官方** `gen_context_qa.py` / `process.py`，
是论文原设计，不是本项目的 bug；7B 能吸收，0.5B 吸收不了。

本脚本在**同一个 adapter** 上跑三种问法，对比 EM：

  A. eval_official  评测原格式：What is the relation between word node X and word node Y?
                    Selected from the following candidate answers: ...
  B. train_ctxqa    训练 context_qa 格式：what is the relation between X and Y?
  C. eval_nocand    评测格式但去掉候选列表（只保留 word node 措辞）

判据
----
* 若 B 或 C 的 EM 显著高于 A（例如从 0.6% → 20%+），则确认根因是**格式错配**，
  而非 "0.5B 记不住图"。
* 若三者都很低（<5%），则是模型容量/训练本身的问题，需重新审查 Stage1/Stage2。

用法（WSL）
----------
cd /mnt/c/Users/Administrator/Desktop/实验/GRIP-CU/13_base_method/grip-exp
python scripts/diagnose_e12_format_mismatch.py \
    --adapter-dir artifacts/nell23k_paper_05b/lora_adapter \
    --num-test 300
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from tqdm.auto import trange

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constants import MODELSCOPE_DECODER_ONLY_LLMS, SYSTEM_PROMPT, TORCH_DTYPE  # noqa: E402
from models.utils import get_hf_llm_tokenizer  # noqa: E402
from utils import extract_tag_content, load_list_json, set_random_seed  # noqa: E402

TEMPLATE = ("Given the context graph titled {title}, please answer the following question: "
            "{question} Response in the following format:<answer>[answer]</answer>")

# 评测官方格式
EVAL_Q = re.compile(r"What is the relation between word node (.+?) and word node (.+?)\? "
                    r"Selected from the following candidate answers: (.+?)\.\s*$")


def build_variants(question: str):
    """把官方评测问句改写成另外两种训练侧问法。"""
    m = EVAL_Q.match(question.strip())
    if not m:
        return None
    src, tgt, cand_str = m.group(1), m.group(2), m.group(3)
    cands = [c.strip() for c in cand_str.split(";")]
    return src, tgt, cands


def make_raw_prompt(tokenizer, title, question):
    """复刻 GRIPEvalDataset 的 chat 构造（含 system prompt + generation prompt）。"""
    # 必须与官方 GRIPEvalDataset._format_chat_template 逐字一致（constants.SYSTEM_PROMPT），
    # 否则诊断出的绝对 EM 会带上额外偏差。
    message = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": TEMPLATE.format(title=title, question=question)},
    ]
    return tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)


# 基座模型缓存目录候选。注意：不能写死 Windows 路径（如 E:/GRIP-CU/model_cache）——
# 在 WSL 下 posixpath.join 会把它拼到 cwd 后面，变成
# "<repo>/E:/GRIP-CU/model_cache" → FileNotFoundError。
MODEL_CACHE_CANDIDATES = (
    'model_cache',                    # 仓库内相对路径（官方脚本默认）
    '/mnt/e/GRIP-CU/model_cache',     # WSL: E 盘
    '/e/GRIP-CU/model_cache',
    'E:/GRIP-CU/model_cache',         # Windows 原生
    'C:/myProgram/GRIP-CU/model_cache',
)


def resolve_model_cache_dir(model_id: str, explicit: str = '') -> str:
    """挑一个真正存在基座模型的缓存目录。"""
    candidates = []
    env = os.environ.get('GRIP_MODEL_CACHE', '').strip()
    if explicit:
        candidates.append(explicit)
    if env:
        candidates.append(env)
    candidates.extend(MODEL_CACHE_CANDIDATES)
    sub = model_id.replace('/', '--')          # Qwen/Qwen2.5-0.5B-Instruct → Qwen--Qwen2.5-0.5B-Instruct
    for cand in candidates:
        d = Path(cand)
        if (d / sub).is_dir():
            return str(d)
    # 都没命中时返回首选，让上游抛出带完整 searched 列表的错误
    return str(candidates[0])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-file', default='outputs/data/nell23k/processed_test.json')
    p.add_argument('--adapter-dir', default='artifacts/nell23k_paper_05b/lora_adapter')
    p.add_argument('--num-test', type=int, default=300)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--model-name', default='qwen-05b')
    p.add_argument('--model-source', default='local')
    p.add_argument('--model-cache-dir', default='',
                   help='留空则自动探测（推荐）。也可用 GRIP_MODEL_CACHE 环境变量覆盖。')
    p.add_argument('--local-files-only', action='store_true', default=True)
    p.add_argument('--gen-max-length', type=int, default=128,
                   help='生成上限。官方评测用 1000；此处默认 128 以把 900 次生成压到'
                        '十几分钟。注意：A 变体的绝对值可能因此略低于官方全量 0.61%，'
                        '但 A/B/C 三者在完全相同的设置下对比，结论不受影响。'
                        '若要逐字复现官方数字，传 1000。')
    p.add_argument('--dump', default='', help='可选：把逐条预测写成 JSONL 便于人工检查')
    return p.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)

    rows = load_list_json(args.input_file)
    rec = rows[0]
    graph, title = rec['graph'], rec['title']
    pairs = []
    for q, tgt in zip(rec['questions'], rec['answers']):
        qtext = q[0] if isinstance(q, list) else q
        parsed = build_variants(qtext)
        if parsed is None:
            continue
        pairs.append((parsed[0], parsed[1], parsed[2], tgt))
    if args.num_test > 0:
        pairs = pairs[:args.num_test]
    print(f'可用样本: {len(pairs)}')

    model_id = MODELSCOPE_DECODER_ONLY_LLMS[args.model_name]
    cache_dir = resolve_model_cache_dir(model_id, args.model_cache_dir)
    if not args.model_cache_dir:
        print(f'自动探测基座模型缓存目录: {cache_dir}')
    model, tokenizer = get_hf_llm_tokenizer(
        model_name=model_id,
        load_dir=str(Path(args.adapter_dir)),
        model_source=args.model_source,
        model_cache_dir=cache_dir,
        local_files_only=args.local_files_only,
        tokenize_max_length=4096,
        dtype=TORCH_DTYPE['bfloat16'],
        peft=False,
        device_map='auto',
    )
    model.eval()

    variants = {
        'A_eval_official': lambda s, t, c: (
            f"What is the relation between word node {s} and word node {t}? "
            f"Selected from the following candidate answers: {'; '.join(c)}."),
        'B_train_ctxqa': lambda s, t, c: f"what is the relation between {s} and {t}?",
        'C_eval_nocand': lambda s, t, c: (
            f"What is the relation between word node {s} and word node {t}?"),
    }

    results = {k: {'em': 0, 'hit': 0, 'in_cand': 0, 'empty': 0} for k in variants}
    dumps = []

    for name, fn in variants.items():
        n_em = n_hit = n_in = n_empty = 0
        for idx in trange(len(pairs), desc=name, unit='q', dynamic_ncols=True):
            src, tgt, cands, answer = pairs[idx]
            question = fn(src, tgt, cands)
            prompt = make_raw_prompt(tokenizer, title, question)
            inputs = tokenizer(prompt, return_tensors='pt')['input_ids'].to(model.device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=args.gen_max_length,
                    do_sample=False,
                )
            text = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
            answers = extract_tag_content(text, 'answer')
            resp = '; '.join(answers).strip()
            if not resp:
                n_empty += 1
            if resp == answer:
                n_em += 1
            if answer in resp:
                n_hit += 1
            if resp in cands:
                n_in += 1
            if args.dump and name == 'A_eval_official':
                dumps.append({'variant': name, 'src': src, 'tgt': tgt, 'target': answer,
                              'response': resp, 'raw': text})
        n = len(pairs)
        results[name] = {'em': 100 * n_em / n, 'hit': 100 * n_hit / n,
                         'in_cand': 100 * n_in / n, 'empty': 100 * n_empty / n}

    print('\n' + '=' * 72)
    print(f'诊断结果（n={len(pairs)}，同一个 adapter）')
    print('=' * 72)
    print(f"{'问法':<18}{'EM%':>9}{'HIT%':>9}{'IN-CAND%':>11}{'EMPTY%':>9}")
    print('-' * 72)
    for name, r in results.items():
        print(f"{name:<18}{r['em']:>9.2f}{r['hit']:>9.2f}{r['in_cand']:>11.2f}{r['empty']:>9.2f}")
    print('-' * 72)
    a = results['A_eval_official']['em']
    b = results['B_train_ctxqa']['em']
    c = results['C_eval_nocand']['em']
    print()
    if max(b, c) > a + 5:
        print('结论：✅ 确认「格式错配」是主因。')
        print(f'     去掉候选列表后 EM 从 {a:.2f}% 提升到 {max(b, c):.2f}%，')
        print('     说明模型记住了图知识，只是不会做 10-way 选择题。')
        print('     修复方向：在训练数据中加入带候选列表的 10-way 问法。')
    elif max(b, c) < 5:
        print('结论：❌ 三种问法都很低 → 不是格式问题，模型根本没记住。')
        print('     需回头审查 Stage1 context memorization / Stage2 QA 训练。')
    else:
        print('结论：⚠️ 有改善但不显著，格式是因素之一，还可能叠加容量/训练问题。')

    if args.dump:
        with open(args.dump, 'w', encoding='utf-8') as f:
            for d in dumps:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')
        print(f'\n逐条预测已写入: {args.dump}')


if __name__ == '__main__':
    main()