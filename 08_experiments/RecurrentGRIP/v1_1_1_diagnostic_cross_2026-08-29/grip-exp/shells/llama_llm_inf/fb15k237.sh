PYTHONPATH=. python scripts/mp_wrapper.py \
    --script scripts/run_llm.py \
    --num_process 1 \
    --dataset fb15k237_2 \
    --output_file fb15k237_llama_baseline_context_submit.json \
    --do_eval \
    --metrics em f1 hit \
    --llm_as_judge_model qwen-32b \
    --report_to_wandb \
    --wandb_project_name grip \
    --wandb_run_name fb15k237_llama_baseline_context_submit \
    --subprocess_args \
    --overwrite True \
    --tokenize_max_length 30000 \
    --gen_max_length 1000 \
    --padding_side left \
    --truncation_side left \
    --model_name llama3-8b \
    --dtype bfloat16 \
    --use_vllm True \
    --batch_size -1 \
    --do_sample True \
    --top_p 0.9 \
    --temperature 0.6 \
    --no_graph_context False \
    --use_subgraph True \
    --index_format False \
    --report_input_token_count True


