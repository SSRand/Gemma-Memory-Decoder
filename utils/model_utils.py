"""
Utility functions for unified access across CausalLM and multimodal (VLM) models.

Multimodal models (e.g., Gemma3ForConditionalGeneration) store text-related config
under `config.text_config`, while pure CausalLM models (e.g., Qwen3ForCausalLM,
Gemma3ForCausalLM) store it directly on `config`. These helpers abstract that away.
"""


def get_text_hidden_size(model):
    """Get text hidden_size from a model, whether CausalLM or multimodal."""
    if hasattr(model.config, 'text_config'):
        return model.config.text_config.hidden_size
    return model.config.hidden_size


def get_text_vocab_size(config):
    """Get text vocab_size from a config object (result of AutoConfig.from_pretrained)."""
    if hasattr(config, 'text_config'):
        return config.text_config.vocab_size
    return config.vocab_size
