from .hf import HFEmbeddingModel


def get_embedding_model(
        model_name: str,
        batch_size: int,
        tokenize_max_length: int,
        **kwargs,
) -> HFEmbeddingModel:
    if model_name.startswith("bge"):
        return HFEmbeddingModel(
            model_name=model_name,
            batch_size=batch_size,
            tokenize_max_length=tokenize_max_length,
        )
    else:
        raise NotImplementedError(f"Embedding model {model_name} is not implemented.")
