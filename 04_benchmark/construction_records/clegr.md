# CLEGR（Facts + Reasoning） 构造记录

- 记录更新时间：`2026-10-21T06:11:22.962830+00:00`
- 构造状态：`ready_complete_low_budget`
- 目标：`组合图问答训练/验证/测试；保留官方 Facts 与 Reasoning 两个变体，合并记录但不混合标签。`

## 构造过程

从本地官方 CLEGR PyTorch archive 逐图转换为 JSONL，保留节点、边、问题、答案、question_type/group/subgroup；按官方 500 图、3:1:1 图级 split 形成 train=300、val=100、test=100。Stage1/Stage2 只从 train 图生成，test 图只用于最终评估。任务类型沿用官方 GripTaskGeneration：context 和未改写的 edge context-QA 由脚本直接构造，summarization/reasoning-QA 由 Qwen-7B 生成；本次是低预算可运行版本，每图 summarization=2、reasoning QA=2、context QA=2，并关闭改写、节点属性 QA 和 context upsampling。它不是官方 CLEGR shell 的完整参数复现；所有模型调用写入 cache。

## 官方 split 与防泄漏检查

图级切分，不是问题级切分；每个 graph_id 在 train/val/test 只出现一次，图内全部问题随图保留。该比例与 CLEGR 官方论文一致；公开 archive 未提供作者隐藏的具体 graph-id 索引，因此 manifest 固定记录 seed=2026 和排序规则。

## 产物元数据

```json
{
  "facts_manifest": {
    "path": "04_benchmark/data/clegr_facts/dataset_manifest.json",
    "size_bytes": 2784,
    "sha256": "ed922366d7b7db03fb07041499b51ca997154e61d0d1161ef9e61365f86bdac9"
  },
  "reasoning_manifest": {
    "path": "04_benchmark/data/clegr_reasoning/dataset_manifest.json",
    "size_bytes": 2828,
    "sha256": "b3d83a21f534e85cfb940c701254552ba3e667305047abe91c2232f963674497"
  },
  "facts_train": {
    "path": "04_benchmark/data/clegr_facts/processed/processed_train.json",
    "size_bytes": 14294763,
    "sha256": "7c04535bd20df930264e670ff5cb50da03185ec90b9897d58fb542616ae25a31"
  },
  "facts_test": {
    "path": "04_benchmark/data/clegr_facts/processed/processed_test.json",
    "size_bytes": 4667844,
    "sha256": "807ace4651a90a2ee5173a7e11f0d8a87abff142f075a4966de41aef35ee4535"
  },
  "reasoning_train": {
    "path": "04_benchmark/data/clegr_reasoning/processed/processed_train.json",
    "size_bytes": 12511418,
    "sha256": "cf95cd591d43a854c53365a06a71c820769691a0db9b4b89c1d9ca82f4fd7e22"
  },
  "reasoning_test": {
    "path": "04_benchmark/data/clegr_reasoning/processed/processed_test.json",
    "size_bytes": 4218863,
    "sha256": "790ac65c12bc1e3fd10a0f698af736f9216432c702ddbe34f4cf62d83f54b16f"
  },
  "facts_tasks": {
    "path": "04_benchmark/data/clegr_facts/tasks/grip_clegr_facts_tasks.json",
    "size_bytes": 46388272,
    "sha256": "3d5ccbe1fff9fce83cc54798cedbf182a08a776104295b5a8540b40ffdc7cb32"
  },
  "reasoning_tasks": {
    "path": "04_benchmark/data/clegr_reasoning/tasks/grip_clegr_reasoning_tasks.json",
    "size_bytes": 47152276,
    "sha256": "22394a3ddf438422ca62dcf1291d4d53803a7e848fcbf9ef3d3b435bc4aeca5e"
  },
  "facts_task_cache": {
    "path": "/mnt/e/GRIP-CU/artifacts/benchmark_tasks/clegr_facts_qwen7b_initial",
    "status": "available",
    "files": {
      "reasoning_qa_s1.jsonl": {
        "records": 600,
        "size_bytes": 2960796
      },
      "reasoning_qa_s2.jsonl": {
        "records": 1192,
        "size_bytes": 5088050
      },
      "summarization_s1.jsonl": {
        "records": 1800,
        "size_bytes": 7054810
      },
      "summarization_s2.jsonl": {
        "records": 1800,
        "size_bytes": 7799320
      }
    }
  },
  "reasoning_task_cache": {
    "path": "/mnt/e/GRIP-CU/artifacts/benchmark_tasks/clegr_reasoning_qwen7b_initial",
    "status": "available",
    "files": {
      "reasoning_qa_s1.jsonl": {
        "records": 600,
        "size_bytes": 2947866
      },
      "reasoning_qa_s2.jsonl": {
        "records": 1193,
        "size_bytes": 5086094
      },
      "summarization_s1.jsonl": {
        "records": 1800,
        "size_bytes": 7100902
      },
      "summarization_s2.jsonl": {
        "records": 1800,
        "size_bytes": 7851770
      }
    }
  }
}
```

## GPU 时间

```json
{
  "model": "Qwen2.5-7B-Instruct, single RTX 3090",
  "initial_budget": "300 train graphs × (2 summaries + 2 reasoning-QA); context QA is deterministic under current flags",
  "official_comparison": {
    "generator_implementation": "same official GripTaskGeneration task classes",
    "current_profile": "summarization=2, reasoning_qa=2, context_qa=2 per train graph; sample_node_attribute_task=False; repharse_context_qa=False; context_upsampling=False",
    "official_clegr_reasoning_shell": "summarization=20, reasoning_qa=160, context_qa=20; default task flags are not overridden",
    "interpretation": "official-generator-compatible low-budget artifact, not an exact official full-budget CLEGR reproduction"
  },
  "prediction": "约 2–8 GPU 小时/变体，合计约 4–16 GPU 小时；首次加载约 3 分钟。若提高到官方 shell 的 20 summaries + 160 reasoning-QA/图，单卡预计会超过 1 天/变体，需另行扩容。",
  "actual": {
    "facts": {
      "status": "complete",
      "completed_at_utc": "2026-10-21T01:41:34.895848+00:00",
      "elapsed_seconds": 14713.004,
      "counts": {
        "context_samples": 27545,
        "qa_samples": 1791,
        "graphs": 300
      },
      "elapsed_hours": 4.087
    },
    "reasoning": {
      "status": "complete",
      "completed_at_utc": "2026-10-21T05:59:46.474679+00:00",
      "elapsed_seconds": 16772.758,
      "counts": {
        "context_samples": 27734,
        "qa_samples": 1792,
        "graphs": 300
      },
      "elapsed_hours": 4.659
    }
  }
}
```

## Stage1 / Stage2 样例

```json
[
  {
    "stage": "summarization_s1",
    "prompt": "You are given several text snippets, which representing real-world facts. Your task is to generate one summary to \n    rephrase all information while exploring potential relationships across the snippets. Specifically:\n    - Be accurate and complete, preserving ALL factual details like color or properties; and numeric details like quantity or coordinate.\n    - rewrite the original context in a completely different way, tone, writing style, and logic to summarize the original context.\n    - Some rewriting techniques can be used, including reordering, exchanging active and passive sentences, or synonym replacement.\n    - If only one text snippet, simply do the summarization and rephrase.\n    - If multiple text snippets are provided, focus on the relations between different text sentences, like multi-entity connections (e.g., triangle relations).\n    The text snippets are provided below:\n    Gnoarm Estate has disabled access and has rail. It features modernist architecture, has clean cleanliness, none music, and is large in size.\n    \n    Please answer in the following format: \nSummary: [summary]. \n    Please DON'T output quotes.",
    "model_output": "Summary: At Gnoarm Estate, visitors with mobility challenges can easily navigate due to its accessible design and presence of rail facilities. The property showcases a contemporary architectural style, maintains high standards of hygiene without any background music, and offers ample space."
  },
  {
    "stage": "summarization_s2",
    "prompt": "You are given several text snippets, which representing real-world facts, along with a summary written from it. \n    Write a new summary that:\n\n    - Retains every correct fact from the provided summary.\n    - Adds any facts the provided summary missed and corrects any inaccuracies, using the text snippets as the source of truth.\n    - Be accurate and complete, preserving ALL factual details like color or properties; and numeric details like quantity or coordinate.\n    - Differs maximally in wording and structure from the provided summary (e.g., reordering, active↔passive, synonym choice).\n    - If the summary is not provided, directly generate a summary from the given text snippets.\n    The text snippets are provided below:\n    Gnoarm Estate has disabled access and has rail. It features modernist architecture, has clean cleanliness, none music, and is large in size.\n    \n    The summary is provided below:\n    At Gnoarm Estate, visitors with mobility challenges can easily navigate due to its accessible design and presence of rail facilities. The property showcases a contemporary architectural style, maintains high standards of hygiene without any background music, and offers ample  …",
    "model_output": "Summary: Gnoarm Estate accommodates individuals with mobility issues through its accessible design and rail amenities. This establishment exhibits a modern architectural aesthetic, upholds strict sanitation protocols free from ambient music, and provides generous room dimensions."
  },
  {
    "stage": "reasoning_qa_s1",
    "prompt": "You are given several text snippets from a graph as context. Generate two diverse, reasoning-focused questions and \n    answers based on the context. \n\n    - The new questions should focus on reasoning over the full subgraph and use >2 distinct facts from the subgraph.\n    - Avoid trivial lookups (e.g., complete an edge) and avoid quantity questions (“how many”).\n    - The answer must include >= 2 element and separate them with comma or \"and\".\n    - Use natural phrasing; do not explicitly states “graph”/“node(s)” in question/answer.\n    - Keep answers concise (single words or short phrases). \n    - The two new questions must target different facts and differ from each other.\n    - Answers must be correct, concise, complete (if multiple elements can be answer, include all of them) and supported by the subgraph.\n\n    text snippets are provided below:\n    the Chuast Court has disabled access and does not have rail. It features modernist architecture, has clean cleanliness, pop music, and is tiny in size. is There is a dashdot purple line from Chuast Court to Criethton. It has air conditioning and was built in 2000. the Criethton has disabled access and has rail. It features modernist  …",
    "model_output": "Evidence: Chuast Court has disabled access and does not have rail. It features modernist architecture, has clean cleanliness, pop music, and is tiny in size. Criethton has disabled access and has rail. It features modernist architecture, has clean cleanliness, pop music, and is small in size. \nQuestion: Which two stations share both modernist architecture and pop music? \nAnswer: Chuast Court, Criethton\n\nEvidence: Ghooswich does not have disabled access and does not have rail. It features art-deco architecture, has dirty cleanliness, rock music, and is large in size. Skeagy Square has disabled access and has rail. It features concrete architecture, has dirty cleanliness, classical music, and is medium-sized in size. \nQuestion: What are the contrasting architectural styles of Ghooswich and Skeagy Square? \nAnswer: art-deco, concrete"
  },
  {
    "stage": "reasoning_qa_s2",
    "prompt": "You are given several text snippets representing real-world facts as context and an QA pair generated from it. Your task \n    is to evaluate the correctness of question and answer pair based on the text snippets. Here is the instruction:\n    1. For each question and answer pair, determine if the question and answer is reasonable and fully grounded by the provided context (answer cannot be I don't know). \n    2. If the question/answer is wrong or not related to the provided context, generate a CORRECTED question/answer based on the context.\n    3. If the answer and question are correct but can be further improved, provide an IMPROVED answer that is more concise and precise.\n    4. If the answer and question already correct and optimal, simply restate the original question and answer.\n        \n    The context and QA pairs are provided below:\n    --Context--\n    the Chuast Court has disabled access and does not have rail. It features modernist architecture, has clean cleanliness, pop music, and is tiny in size. is There is a dashdot purple line from Chuast Court to Criethton. It has air conditioning and was built in 2000. the Criethton has disabled access and has rail. It features mod …",
    "model_output": "Changed: no \nQuestion: Which two stations share both modernist architecture and pop music? \nAnswer: Chuast Court, Criethton"
  },
  {
    "stage": "summarization_s1",
    "prompt": "You are given several text snippets, which representing real-world facts. Your task is to generate one summary to \n    rephrase all information while exploring potential relationships across the snippets. Specifically:\n    - Be accurate and complete, preserving ALL factual details like color or properties; and numeric details like quantity or coordinate.\n    - rewrite the original context in a completely different way, tone, writing style, and logic to summarize the original context.\n    - Some rewriting techniques can be used, including reordering, exchanging active and passive sentences, or synonym replacement.\n    - If only one text snippet, simply do the summarization and rephrase.\n    - If multiple text snippets are provided, focus on the relations between different text sentences, like multi-entity connections (e.g., triangle relations).\n    The text snippets are provided below:\n    Graiotston does not have disabled access and has rail. It features glass architecture, has clean cleanliness, electronic music, and is large in size.\n    \n    Please answer in the following format: \nSummary: [summary]. \n    Please DON'T output quotes.",
    "model_output": "Summary: While Graiotston lacks facilities for disabled access, it boasts rail transportation. The venue is known for its sleek glass design, maintains high standards of cleanliness, hosts electronic music events, and is notably spacious."
  },
  {
    "stage": "summarization_s2",
    "prompt": "You are given several text snippets, which representing real-world facts, along with a summary written from it. \n    Write a new summary that:\n\n    - Retains every correct fact from the provided summary.\n    - Adds any facts the provided summary missed and corrects any inaccuracies, using the text snippets as the source of truth.\n    - Be accurate and complete, preserving ALL factual details like color or properties; and numeric details like quantity or coordinate.\n    - Differs maximally in wording and structure from the provided summary (e.g., reordering, active↔passive, synonym choice).\n    - If the summary is not provided, directly generate a summary from the given text snippets.\n    The text snippets are provided below:\n    Graiotston does not have disabled access and has rail. It features glass architecture, has clean cleanliness, electronic music, and is large in size.\n    \n    The summary is provided below:\n    While Graiotston lacks facilities for disabled access, it boasts rail transportation. The venue is known for its sleek glass design, maintains high standards of cleanliness, hosts electronic music events, and is notably spacious.\n\n    Please answer in the following f …",
    "model_output": "Summary: Graiotston does not offer accessibility for individuals with disabilities but provides rail transport. Its architecture is characterized by glass elements, it keeps a high level of cleanliness, frequently hosts electronic music concerts, and is notably large in size."
  },
  {
    "stage": "reasoning_qa_s1",
    "prompt": "You are given several text snippets from a graph as context. Generate two diverse, reasoning-focused questions and \n    answers based on the context. \n\n    - The new questions should focus on reasoning over the full subgraph and use >2 distinct facts from the subgraph.\n    - Avoid trivial lookups (e.g., complete an edge) and avoid quantity questions (“how many”).\n    - The answer must include >= 2 element and separate them with comma or \"and\".\n    - Use natural phrasing; do not explicitly states “graph”/“node(s)” in question/answer.\n    - Keep answers concise (single words or short phrases). \n    - The two new questions must target different facts and differ from each other.\n    - Answers must be correct, concise, complete (if multiple elements can be answer, include all of them) and supported by the subgraph.\n\n    text snippets are provided below:\n    the Xafy Palace has disabled access and has rail. It features modernist architecture, has dirty cleanliness, pop music, and is medium-sized in size. is There is a dashdot pink line from Shrupp Boulevard to Xafy Palace. It has air conditioning and was built in 2010. the Shrupp Boulevard does not have disabled access and has rail. It f …",
    "model_output": "Evidence: Xafy Palace has modernist architecture and is medium-sized, while Prountlyington has victorian architecture and is large in size. \nQuestion: Which palace is larger in size? \nAnswer: Prountlyington\n\nEvidence: There is a dashdot pink line from Shrupp Boulevard to Xafy Palace and another dashdot pink line from Pruwneyland to Kiefy Court. \nQuestion: Which two locations are connected by a dashdot pink line? \nAnswer: Shrupp Boulevard, Xafy Palace and Pruwneyland, Kiefy Court"
  },
  {
    "stage": "reasoning_qa_s2",
    "prompt": "You are given several text snippets representing real-world facts as context and an QA pair generated from it. Your task \n    is to evaluate the correctness of question and answer pair based on the text snippets. Here is the instruction:\n    1. For each question and answer pair, determine if the question and answer is reasonable and fully grounded by the provided context (answer cannot be I don't know). \n    2. If the question/answer is wrong or not related to the provided context, generate a CORRECTED question/answer based on the context.\n    3. If the answer and question are correct but can be further improved, provide an IMPROVED answer that is more concise and precise.\n    4. If the answer and question already correct and optimal, simply restate the original question and answer.\n        \n    The context and QA pairs are provided below:\n    --Context--\n    the Xafy Palace has disabled access and has rail. It features modernist architecture, has dirty cleanliness, pop music, and is medium-sized in size. is There is a dashdot pink line from Shrupp Boulevard to Xafy Palace. It has air conditioning and was built in 2010. the Shrupp Boulevard does not have disabled access and has rai …",
    "model_output": "Changed: yes \nQuestion: Which palace is larger in size? \nAnswer: Prountlyington"
  }
]
```

## 下游结论

PASS（低预算版本）：两个变体的 generation_status 均为 complete；原始数据与图级 split 可用于下游，但不能将当前任务文件宣称为官方 CLEGR 完整预算复现。训练/验证/测试不得按问题随机打散跨图。
