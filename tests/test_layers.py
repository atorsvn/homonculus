import pytest
import tensorflow as tf
import keras
from lora_h2_bart.layers import VectorQuantizer, LoRADense, HomuncularController
from lora_h2_bart.config import ModelConfig

class TestLayers:
    
    def setup_method(self):
        self.config = ModelConfig(hidden_dim=32, num_adapters=4, rl_action_space_size=8)
        self.batch, self.seq, self.dim = 2, 10, 32

    def test_vq_shape_and_ste_gradients(self):
        """Test if VQ preserves shape and passes gradients via STE."""
        vq = VectorQuantizer(num_embeddings=50, embedding_dim=self.dim)
        x = tf.random.normal((self.batch, self.seq, self.dim))
        
        with tf.GradientTape() as tape:
            tape.watch(x)
            out = vq(x)
            loss = tf.reduce_mean(out**2)
            
        assert out.shape == (self.batch, self.seq, self.dim)
        grad = tape.gradient(loss, x)
        assert grad is not None
        assert tf.reduce_sum(tf.abs(grad)) > 0

    def test_lora_orthogonality(self):
        """Test if orthogonality loss is calculable and non-negative."""
        dense = keras.layers.Dense(self.dim)
        dense.build((None, self.dim))
        lora = LoRADense(dense, rank=4, num_adapters=self.config.num_adapters)
        lora.build((None, self.dim))
        
        loss = lora.orthogonality_loss()
        assert loss.shape == () # Scalar
        assert loss >= 0

    def test_rl_agent_outputs(self):
        """Test Homuncular Controller for RL specific outputs."""
        agent = HomuncularController(config=self.config)
        x = tf.random.normal((self.batch, self.seq, self.dim))
        
        out = agent(x)
        assert "steer" in out
        assert "value" in out
        assert "action_logits" in out
        
        # Steer should be [B, 1, Dim] 
        assert out["steer"].shape == (self.batch, 1, self.dim)
        # Value should be [B, 1]
        assert out["value"].shape == (self.batch, 1)
        # Action Logits should be [B, ActionSpaceSize]
        assert out["action_logits"].shape == (self.batch, self.config.rl_action_space_size)