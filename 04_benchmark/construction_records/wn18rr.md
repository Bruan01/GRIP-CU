# WN18RR 构造记录

- 记录更新时间：`2026-10-21T06:11:17.943635+00:00`
- 构造状态：`ready_complete`
- 目标：`按 GRIP 官方单图协议从 train 图构造 Stage1/Stage2 任务，并用 valid/test 做关系预测评估。`

## 构造过程

原始 entity2text/train/valid/test 从已复现 GRIP 资产复制并做 SHA256 登记；processed_test/processed_val 保留完整 train 图作为上下文，问题分别来自官方 test/valid。任务类型沿用官方 GripTaskGeneration：context 和未改写的 edge context-QA 由脚本直接构造，summarization/reasoning-QA 由 Qwen-7B 生成。配置为 context=8000、reasoning QA=2000、summarization=6000，关闭节点属性 QA 和 context QA 改写；每个模型阶段写入可恢复 JSONL cache。

## 官方 split 与防泄漏检查

train/valid/test 原始三分区互不相交；所有图上下文边来自 train.txt，valid/test 只提供问题与答案。WN 没有伪造 processed_train：参考 QA 使用官方 valid，这与原 GRIP WN18RR 处理器和 mp_wrapper 的 fallback 一致。

## 产物元数据

```json
{
  "manifest": {
    "path": "04_benchmark/data/wn18rr/manifest.json",
    "size_bytes": 2225,
    "sha256": "591df8d5529430ffb596eb8b455fd9530d80c164ce502a1da1e9d40d7325157a"
  },
  "processed_val": {
    "path": "04_benchmark/data/wn18rr/processed/processed_val.json",
    "size_bytes": 25873662,
    "sha256": "0977646ae09739f6155cbeccf7fb8bedaf3009c2098e2c2f062bb025e93bff3e"
  },
  "processed_test": {
    "path": "04_benchmark/data/wn18rr/processed/processed_test.json",
    "size_bytes": 25915687,
    "sha256": "0156364b9901aba7c7a1f15a82af7af2d40990457a8baf87a14c224c57e548e0"
  },
  "generation_plan": {
    "path": "04_benchmark/data/wn18rr/tasks/generation_plan.json",
    "size_bytes": 645,
    "sha256": "94b917230f3cc3cd88380a535b51d85da79ac1fceacff85808dd4801e7c73921"
  },
  "tasks": {
    "path": "04_benchmark/data/wn18rr/tasks/grip_wn18rr_tasks.json",
    "size_bytes": 201699990,
    "sha256": "feb7c3621119666b60c91f52a1b70a0b37f0bf9610ab283664cc253384c2db12"
  },
  "task_cache": {
    "path": "/mnt/e/GRIP-CU/artifacts/benchmark_tasks/wn18rr_qwen7b_official",
    "status": "available",
    "files": {
      "reasoning_qa_s1.jsonl": {
        "records": 2000,
        "size_bytes": 9567043
      },
      "reasoning_qa_s2.jsonl": {
        "records": 4336,
        "size_bytes": 13067798
      },
      "summarization_s1.jsonl": {
        "records": 18000,
        "size_bytes": 46392963
      },
      "summarization_s2.jsonl": {
        "records": 18000,
        "size_bytes": 51923305
      }
    }
  }
}
```

## GPU 时间

```json
{
  "calibration": "Qwen2.5-7B 首次加载实测约 163 秒、显存约 14.2 GiB；一图一例 smoke 已成功完成。",
  "running_configuration": "单 RTX 3090，guardenv，batch=4；cache 可断点续跑。",
  "prediction": "按正式配置约 42,000 个模型 prompt、当前 batch=4 实测约 2–4 秒/prompt，预计约 20–60 GPU 小时（I/O、输出长度和 cache 命中会改变上界）；首次模型加载另计约 3 分钟。",
  "actual": {
    "status": "complete",
    "completed_at_utc": null,
    "elapsed_seconds": null,
    "counts": {
      "context_samples": 163662,
      "qa_samples": 12334,
      "graphs": 1
    }
  }
}
```

## Stage1 / Stage2 样例

```json
[
  {
    "stage": "summarization_s1",
    "prompt": "You are given several text snippets, which representing real-world facts. Your task is to generate one summary to \n    rephrase all information while exploring potential relationships across the snippets. Specifically:\n    - Be accurate and complete, preserving ALL factual details like color or properties; and numeric details like quantity or coordinate.\n    - rewrite the original context in a completely different way, tone, writing style, and logic to summarize the original context.\n    - Some rewriting techniques can be used, including reordering, exchanging active and passive sentences, or synonym replacement.\n    - If only one text snippet, simply do the summarization and rephrase.\n    - If multiple text snippets are provided, focus on the relations between different text sentences, like multi-entity connections (e.g., triangle relations).\n    The text snippets are provided below:\n    unworthiness, the quality of being not particularly suitable or befitting; \"he retracted nothing that he had said about the inappropriateness of either a corporeal God or a God who is a person\"; \"his praise released from her loud protestations of her unworthiness\"\n    \n    Please answer in the fol …",
    "model_output": "Summary: The concept of unworthiness was central, indicating a lack of suitability for something sacred or significant. This theme was evident in statements rejecting the notion of a physical deity or a personal god. Conversely, it also surfaced in a scenario where someone's commendation alleviated another's self-doubt regarding their fitness for a divine connection."
  },
  {
    "stage": "summarization_s2",
    "prompt": "You are given several text snippets, which representing real-world facts, along with a summary written from it. \n    Write a new summary that:\n\n    - Retains every correct fact from the provided summary.\n    - Adds any facts the provided summary missed and corrects any inaccuracies, using the text snippets as the source of truth.\n    - Be accurate and complete, preserving ALL factual details like color or properties; and numeric details like quantity or coordinate.\n    - Differs maximally in wording and structure from the provided summary (e.g., reordering, active↔passive, synonym choice).\n    - If the summary is not provided, directly generate a summary from the given text snippets.\n    The text snippets are provided below:\n    unworthiness, the quality of being not particularly suitable or befitting; \"he retracted nothing that he had said about the inappropriateness of either a corporeal God or a God who is a person\"; \"his praise released from her loud protestations of her unworthiness\"\n    \n    The summary is provided below:\n    The concept of unworthiness was central, indicating a lack of suitability for something sacred or significant. This theme was evident in statements reje …",
    "model_output": "Summary: The idea of unworthiness played a key role, signifying a deficiency in appropriateness for sacred or significant matters. This theme appeared in assertions against the existence of a tangible deity or a god conceived as an individual entity. On the other hand, it also emerged in instances where accolades lessened another’s reservations about their readiness for a spiritual bond."
  },
  {
    "stage": "reasoning_qa_s1",
    "prompt": "You are given several text snippets from a graph as context. Generate two diverse, reasoning-focused questions and \n    answers based on the context. \n\n    - The new questions should focus on reasoning over the full subgraph and use >2 distinct facts from the subgraph.\n    - Avoid trivial lookups (e.g., complete an edge) and avoid quantity questions (“how many”).\n    - The answer must include >= 2 element and separate them with comma or \"and\".\n    - Use natural phrasing; do not explicitly states “graph”/“node(s)” in question/answer.\n    - Keep answers concise (single words or short phrases). \n    - The two new questions must target different facts and differ from each other.\n    - Answers must be correct, concise, complete (if multiple elements can be answer, include all of them) and supported by the subgraph.\n\n    text snippets are provided below:\n    the mandarin orange, a somewhat flat reddish-orange loose skinned citrus of China is hypernym the citrus fruit, any of numerous fruits of the genus Citrus having thick rind and juicy pulp; grown in warm regions; the worst, (superlative of `bad') most wanting in quality or value or condition; \"the worst player on the team\"; \"the worst …",
    "model_output": "Evidence: the mandarin orange is a citrus fruit, and the citrus fruit is grown in warm regions.\nQuestion: Where might you find a mandarin orange growing?\nAnswer: warm regions\n\nEvidence: the worst is the superlative of bad, and drop is derivationally related to degenerate.\nQuestion: What term related to quality would be considered the most extreme form of bad?\nAnswer: the worst\n\nEvidence: the thread is used to string pearls, and thread is used to run or pass over something.\nQuestion: When threading a needle, what action are you performing?\nAnswer: running\n\nEvidence: the impairment is a symptom of reduced quality, and drop means to grow worse.\nQuestion: How might an impairment manifest itself?\nAnswer: growing worse\n\nEvidence: the committee is a group, and the mandarin orange is a type of citrus fruit.\nQuestion: What does a committee have in common with a mandarin orange?\nAnswer: both are types of groups\n\nEvidence: the tame is a process of making something docile, and the mandarin orange is a citrus fruit.\nQuestion: What process might be applied to a citrus fruit to alter its nature?\nAnswer: taming"
  },
  {
    "stage": "reasoning_qa_s2",
    "prompt": "You are given several text snippets representing real-world facts as context and an QA pair generated from it. Your task \n    is to evaluate the correctness of question and answer pair based on the text snippets. Here is the instruction:\n    1. For each question and answer pair, determine if the question and answer is reasonable and fully grounded by the provided context (answer cannot be I don't know). \n    2. If the question/answer is wrong or not related to the provided context, generate a CORRECTED question/answer based on the context.\n    3. If the answer and question are correct but can be further improved, provide an IMPROVED answer that is more concise and precise.\n    4. If the answer and question already correct and optimal, simply restate the original question and answer.\n        \n    The context and QA pairs are provided below:\n    --Context--\n    the mandarin orange, a somewhat flat reddish-orange loose skinned citrus of China is hypernym the citrus fruit, any of numerous fruits of the genus Citrus having thick rind and juicy pulp; grown in warm regions; the worst, (superlative of `bad') most wanting in quality or value or condition; \"the worst player on the team\"; \"th …",
    "model_output": "Changed: yes \nQuestion: Where are citrus fruits typically grown? \nAnswer: warm regions"
  }
]
```

## 下游结论

PASS：原始 split、上下文图和参考 QA 没有测试泄漏；任务文件已完成，可进入下游训练/评估。
