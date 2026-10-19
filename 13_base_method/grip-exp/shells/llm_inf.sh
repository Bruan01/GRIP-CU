PYTHONPATH=. python scripts/mp_wrapper.py \
    --script scripts/run_llm.py \
    --num_process 8 \
    --dataset clegr_reasoning \
    --num_test 400 \
    --output_file clegr_reasoning_llama_baseline_rerun.json \
    --do_eval \
    --metrics em f1 hit llm\
    --llm_as_judge_model qwen-32b \
    --report_to_wandb \
    --wandb_project_name grip \
    --wandb_run_name clegr_reasoning_llama_baseline_rerun \
    --subprocess_args \
    --overwrite True\
    --tokenize_max_length 30000 \
    --gen_max_length 1000 \
    --model_name llama3-8b \
    --dtype bfloat16 \
    --use_vllm True \
    --do_sample True \
    --top_p 0.9 \
    --temperature 0.6 \
    --no_graph_context False \
    --use_subgraph False \


