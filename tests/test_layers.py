import pytest
import tensorflow as tf
import keras
from lora_h2_bart.layers import VectorQuantizer, LoRADense, HomuncularController

class TestLayers:
    
    def test_vq_shape_and_gradients(self):
        """Test if VQ preserves shape and passes gradients via STE."""
        batch, seq, dim = 2, 10, 32
        vq = VectorQuantizer(num_embeddings=50, embedding_dim=dim)
        x = tf.random.normal((batch, seq, dim))
        
        with tf.GradientTape() as tape:
            tape.watch(x)
            out = vq(x)
            loss = tf.reduce_mean(out**2)
            
        assert out.shape == (batch, seq, dim)
        # Check if gradient flows back to input (STE check)
        grad = tape.gradient(loss, x)
        assert grad is not None
        assert tf.reduce_sum(tf.abs(grad)) > 0

    def test_lora_orthogonality(self):
        """Test if orthogonality loss is calculable."""
        dense = keras.layers.Dense(32)
        dense.build((None, 32))
        lora = LoRADense(dense, rank=4, num_adapters=3)
        lora.build((None, 32))
        
        loss = lora.orthogonality_loss()
        assert loss.shape == () # Scalar
        assert loss >= 0

    def test_agent_outputs(self):
        """Test Homuncular Controller output dictionary."""
        batch, seq, dim = 2, 10, 32
        agent = HomuncularController(hidden_dim=dim, num_adapters=5)
        x = tf.random.normal((batch, seq, dim))
        
        out = agent(x)
        assert "route" in out
        assert "steer" in out
        assert "ponder" in out
        
        # Route should be [B, Num_Adapters]
        assert out["route"].shape == (batch, 5)
        # Steer should be [B, 1, Dim] (Broadcast ready)
        assert out["steer"].shape == (batch, 1, dim)