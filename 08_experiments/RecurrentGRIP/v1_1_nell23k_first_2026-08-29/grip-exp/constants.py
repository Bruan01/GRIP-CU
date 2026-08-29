import torch

HF_DECODER_ONLY_LLMS = {
    "llama3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "llama3-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "Mistral-8b": "mistralai/Ministral-8B-Instruct-2410",
    "qwen-32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
}

# ModelScope mirrors. Prefer these from mainland China before falling back to Hugging Face.
MODELSCOPE_DECODER_ONLY_LLMS = {
    "llama3-70b": "LLM-Research/Meta-Llama-3.3-70B-Instruct",
    "llama3-8b": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
    "Mistral-8b": "AI-ModelScope/Ministral-8B-Instruct-2410",
    "qwen-32b": "Qwen/Qwen2.5-32B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
}

HF_EMBEDDING_LLMS = {
    "bge_large": "BAAI/bge-large-en-v1.5",
    "bge_small": "BAAI/bge-small-en-v1.5",
    "bge_m3": "BAAI/bge-m3"
}

TORCH_DTYPE = {
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "float32": torch.float32,
    "float16": torch.float16,
    "int": torch.int,
    "long": torch.long,
    "bool": torch.bool,
}

NODE_TAG = "node"
EDGE_TAG = "rel"
GRAPH_TAG = "graph"
ANSWER_TAG = "answer"
EVIDENCE_TAG = "evidence"
QUESTION_TAG = "question"
# GRAPH_PLACEHOLDER = "{{graph}}"

TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"

SYSTEM_PROMPT = """
You are a helpful assistant. your task is to provide accurate, direct, and concise answers to user queries. If unable to 
answer the question, respond with "I don't know". Do not guess the answer.
"""

