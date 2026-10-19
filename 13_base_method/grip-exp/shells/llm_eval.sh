PYTHONPATH=. python scripts/run_eval.py \
    --dataset_name clegr_facts \
    --input_file outputs/grip_inf/clegr_facts/clegr_facts_qwen_submit.json \
    --output_file outputs/grip_inf/clegr_facts/clegr_facts_qwen_submit_wrong.json \
    --llm_as_judge_model qwen-32b \
    --tokenize_max_length 30000 \
    --gen_max_length 1000 \


