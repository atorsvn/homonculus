import pytest
import tensorflow as tf
from lora_h2_bart.model import LoRA_H2_BART
from lora_h2_bart.config import ModelConfig

def test_full_model_forward_pass():
    """Test the complete assembly of the automaton."""
    config = ModelConfig(
        hidden_dim=768,
        num_adapters=3,
        vq_codebook_size=128
    )
    
    model = LoRA_H2_BART(config)
    
    # Dummy Input (BART Token IDs)
    inputs = {
        "token_ids": tf.ones((2, 50), dtype=tf.int32),
        "padding_mask": tf.ones((2, 50), dtype=tf.int32)
    }
    
    # Run Inference
    memory, agent_out = model(inputs)
    
    # Check Shapes
    assert memory.shape == (2, 50, 768)
    assert agent_out["route"].shape == (2, 3)