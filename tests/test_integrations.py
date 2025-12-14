import pytest
import tensorflow as tf
from lora_h2_bart.model import LoRA_H2_BART_RL
from lora_h2_bart.config import ModelConfig

def test_full_model_forward_pass_rl():
    """Test the complete assembly of the RL automaton and Vector DB query."""
    config = ModelConfig(
        hidden_dim=768,
        num_adapters=3,
        vq_codebook_size=128,
        rl_action_space_size=10
    )
    
    model = LoRA_H2_BART_RL(config)
    
    # Dummy Input (BART Token IDs)
    inputs = {
        "token_ids": tf.ones((1, 50), dtype=tf.int32),
        "padding_mask": tf.ones((1, 50), dtype=tf.int32)
    }
    
    # Run Inference
    memory, agent_out, task_idx = model(inputs)
    
    # Check Shapes
    assert memory.shape == (1, 50, 768)
    assert agent_out["action_logits"].shape == (1, 10)
    assert agent_out["value"].shape == (1, 1)
    
    # Check Router Output (should be a single index)
    assert task_idx.shape == ()
    assert task_idx < config.num_adapters