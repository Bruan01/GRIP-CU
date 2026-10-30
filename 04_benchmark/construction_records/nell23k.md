# NELL23K 构造记录

- 记录更新时间：`2026-10-21T06:11:01.666547+00:00`
- 构造状态：`ready_existing_task_artifact`
- 目标：`关系预测训练/验证/测试；复用已经复现并完成的 GRIP 任务构造。`

## 构造过程

保留官方 train/valid/test 三个原始关系文件；使用现有 GRIP processed_val/processed_test 和 Qwen-7B 任务文件。此次只做校验和登记，没有重复调用模型。Stage1/Stage2 的原始 Qwen 缓存位于 `/mnt/e/GRIP-CU/tasks/cache`。

## 官方 split 与防泄漏检查

原始三分区互不相交；训练图只来自 train.txt，valid/test 仅用于评估。NELL 的 processed 文件沿用已复现入口，因此保留其原始处理器的兼容性行为。

## 产物元数据

```json
{
  "manifest": {
    "path": "04_benchmark/data/nell23k/manifest.json",
    "size_bytes": 2021,
    "sha256": "e9c2c5535db0da99919662a3b03b8ced197d8198af2d9b4d8af882abe762de9b"
  },
  "processed_val": {
    "path": "04_benchmark/data/nell23k/processed/processed_val.json",
    "size_bytes": 5921580,
    "sha256": "cd775294ba400f166f295cd98d1a8237002875d2487c3de1b46989aa4fe0d2c8"
  },
  "processed_test": {
    "path": "04_benchmark/data/nell23k/processed/processed_test.json",
    "size_bytes": 5920351,
    "sha256": "7c423e5a806540a1355b56e0787e2948e8a899c789daa6d5a5464dfff5124425"
  },
  "tasks": {
    "path": "04_benchmark/data/nell23k/tasks/grip_nell23k_tasks.json",
    "size_bytes": 88081693,
    "sha256": "75f804181261c2bd00e67315fe1cb35d409a1e8a6b0862708627f3f987d3066d"
  },
  "task_cache": {
    "path": "/mnt/e/GRIP-CU/tasks/cache",
    "status": "available",
    "files": {
      "reasoning_qa_s1.jsonl": {
        "records": 2010,
        "size_bytes": 7541049
      },
      "reasoning_qa_s2.jsonl": {
        "records": 4046,
        "size_bytes": 9001433
      },
      "summarization_s1.jsonl": {
        "records": 18060,
        "size_bytes": 29696416
      },
      "summarization_s2.jsonl": {
        "records": 18060,
        "size_bytes": 31443413
      }
    }
  }
}
```

## GPU 时间

```json
{
  "this_run_gpu_seconds": 0,
  "full_task_generation": "已有产物，本轮不重复推理",
  "model": "Qwen2.5-7B-Instruct (登记来源；本轮未调用)"
}
```

## Stage1 / Stage2 样例

```json
[
  {
    "stage": "summarization_s1",
    "prompt": "You are given several text snippets, which representing real-world facts. Your task is to generate one summary to \n    rephrase all information while exploring potential relationships across the snippets. Specifically:\n    - Be accurate and complete, preserving ALL factual details like color or properties; and numeric details like quantity or coordinate.\n    - rewrite the original context in a completely different way, tone, writing style, and logic to summarize the original context.\n    - Some rewriting techniques can be used, including reordering, exchanging active and passive sentences, or synonym replacement.\n    - If only one text snippet, simply do the summarization and rephrase.\n    - If multiple text snippets are provided, focus on the relations between different text sentences, like multi-entity connections (e.g., triangle relations).\n    The text snippets are provided below:\n    concept_island_houston\n    \n    Please answer in the following format: \nSummary: [summary]. \n    Please DON'T output quotes.",
    "model_output": "Summary: In Houston, there exists a concept known as Concept Island, an imaginary or theoretical area that serves as a metaphorical representation for various ideas or principles within the city's urban landscape. This concept highlights the diverse and complex nature of Houston's cultural, economic, and social structures, illustrating how different elements interconnect and influence each other."
  },
  {
    "stage": "summarization_s2",
    "prompt": "You are given several text snippets, which representing real-world facts, along with a summary written from it. \n    Write a new summary that:\n\n    - Retains every correct fact from the provided summary.\n    - Adds any facts the provided summary missed and corrects any inaccuracies, using the text snippets as the source of truth.\n    - Be accurate and complete, preserving ALL factual details like color or properties; and numeric details like quantity or coordinate.\n    - Differs maximally in wording and structure from the provided summary (e.g., reordering, active↔passive, synonym choice).\n    - If the summary is not provided, directly generate a summary from the given text snippets.\n    The text snippets are provided below:\n    concept_island_houston\n    \n    The summary is provided below:\n    In Houston, there exists a concept known as Concept Island, an imaginary or theoretical area that serves as a metaphorical representation for various ideas or principles within the city's urban landscape. This concept highlights the diverse and complex nature of Houston's cultural, economic, and social structures, illustrating how different elements interconnect and influence each other.\n\n   …",
    "model_output": "Summary: Concept Island symbolizes a conceptual space in Houston that illustrates the intricate connections among the city’s cultural, economic, and social dimensions. This metaphor underscores the multifaceted and interconnected nature of Houston's urban fabric, showcasing how various elements interact and shape one another."
  },
  {
    "stage": "reasoning_qa_s1",
    "prompt": "You are given several text snippets from a graph as context. Generate two diverse, reasoning-focused questions and \n    answers based on the context. \n\n    - The new questions should focus on reasoning over the full subgraph and use >2 distinct facts from the subgraph.\n    - Avoid trivial lookups (e.g., complete an edge) and avoid quantity questions (“how many”).\n    - The answer must include >= 2 element and separate them with comma or \"and\".\n    - Use natural phrasing; do not explicitly states “graph”/“node(s)” in question/answer.\n    - Keep answers concise (single words or short phrases). \n    - The two new questions must target different facts and differ from each other.\n    - Answers must be correct, concise, complete (if multiple elements can be answer, include all of them) and supported by the subgraph.\n\n    text snippets are provided below:\n    the concept_airport_bwi is concept:buildinglocatedincity the concept_city_baltimore; the concept_bacteria_mycobacterium_marinum is concept:bacteriaisthecausativeagentofphysiologicalcondition the concept_physiologicalcondition_infection; the concept_politicianus_president_bill_clinton is concept:politicianusendorsedbypoliticianus the  …",
    "model_output": "Evidence: the concept_airport_bwi is concept:buildinglocatedincity the concept_city_baltimore; the concept_politicianus_president_bill_clinton is concept:politicianusendorsedbypoliticianus the concept_politicianus_george_w__b__ush; \nQuestion: Who endorsed Bill Clinton? \nAnswer: George W. Bush, President Bush\n\nEvidence: the concept_bacteria_mycobacterium_marinum is concept:bacteriaisthecausativeagentofphysiologicalcondition the concept_physiologicalcondition_infection; the concept_arthropod_raccoons is concept:animalsuchasinvertebrate the concept_animal_mammals001; \nQuestion: What type of bacteria causes infection? \nAnswer: Mycobacterium marinum, Bacteria"
  },
  {
    "stage": "reasoning_qa_s2",
    "prompt": "You are given several text snippets representing real-world facts as context and an QA pair generated from it. Your task \n    is to evaluate the correctness of question and answer pair based on the text snippets. Here is the instruction:\n    1. For each question and answer pair, determine if the question and answer is reasonable and fully grounded by the provided context (answer cannot be I don't know). \n    2. If the question/answer is wrong or not related to the provided context, generate a CORRECTED question/answer based on the context.\n    3. If the answer and question are correct but can be further improved, provide an IMPROVED answer that is more concise and precise.\n    4. If the answer and question already correct and optimal, simply restate the original question and answer.\n        \n    The context and QA pairs are provided below:\n    --Context--\n    the concept_airport_bwi is concept:buildinglocatedincity the concept_city_baltimore; the concept_bacteria_mycobacterium_marinum is concept:bacteriaisthecausativeagentofphysiologicalcondition the concept_physiologicalcondition_infection; the concept_politicianus_president_bill_clinton is concept:politicianusendorsedbypolitician …",
    "model_output": "Changed: yes \nQuestion: Who endorsed Bill Clinton according to the given context? \nAnswer: George W. Bush"
  }
]
```

## 下游结论

PASS：任务文件、processed 文件和原始 split 均已存在，可直接进入下游训练/评估；测试答案仍来自官方测试文件，不从合成任务回填。
