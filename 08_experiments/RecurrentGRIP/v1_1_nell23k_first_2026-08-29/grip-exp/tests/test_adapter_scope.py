import unittest

try:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
except ImportError:  # pragma: no cover - project runtime installs these dependencies.
    LoraConfig = TaskType = get_peft_model = LlamaConfig = LlamaForCausalLM = None

from grip.recurrent.executor import wrap_decoder_layer
from grip.recurrent.model import validate_adapter_scope


@unittest.skipIf(LlamaConfig is None, "transformers and peft are required")
class AdapterScopeTest(unittest.TestCase):
    def test_peft_layer_filter_matches_wrapped_executor_path(self):
        base = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=4,
            )
        )
        resolved = wrap_decoder_layer(base, layer_index=1, depth=2)
        model = get_peft_model(
            base,
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["q_proj", "v_proj"],
                layers_to_transform=[resolved],
                layers_pattern="layers",
                task_type=TaskType.CAUSAL_LM,
            ),
        )
        names = validate_adapter_scope(model, resolved)
        self.assertTrue(names)
        self.assertTrue(all(".layers.1.block." in name for name in names))


if __name__ == "__main__":
    unittest.main()
