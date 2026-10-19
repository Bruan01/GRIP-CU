PYTHONPATH=. python scripts/mp_wrapper.py \
    --script scripts/run_llm.py \
    --num_process 8 \
    --dataset scene_graph \
    --num_test 500 \
    --output_file scene_graph_qwen_baseline_context_submit.json \
    --do_eval \
    --metrics em f1 hit llm \
    --llm_as_judge_model qwen-32b \
    --report_to_wandb \
    --wandb_project_name grip \
    --wandb_run_name scene_graph_qwen_baseline_context_submit \
    --subprocess_args \
    --overwrite True\
    --tokenize_max_length 30000 \
    --gen_max_length 1000 \
    --model_name qwen-7b \
    --dtype bfloat16 \
    --use_vllm True \
    --batch_size -1 \
    --do_sample True \
    --top_p 0.9 \
    --temperature 0.6 \
    --no_graph_context False \
    --use_subgraph False \
    --index_format False \
    --report_input_token_count True


