import keras
import keras_nlp
import tensorflow as tf
from .layers import LoRADense, ResidualVQ, HomuncularController
from .config import ModelConfig

class LoRA_H2_BART(keras.Model):
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # 1. Backbone
        self.backbone = keras_nlp.models.BartBackbone.from_preset(config.preset)
        
        # 2. Plasticity (Mock injection for demonstration)
        # Real implementation would recurse backbone layers
        self.lora_proj = LoRADense(
            keras.layers.Dense(config.hidden_dim),
            rank=config.lora_rank,
            num_adapters=config.num_adapters
        )

        # 3. Stability
        self.vq = ResidualVQ(
            num_quantizers=config.num_quantizers,
            num_embeddings=config.vq_codebook_size,
            embedding_dim=config.hidden_dim
        )
        
        # 4. Agency
        self.agent = HomuncularController(
            hidden_dim=config.hidden_dim,
            num_adapters=config.num_adapters
        )
        
        self.ortho_tracker = keras.metrics.Mean(name="ortho_loss")

    def call(self, inputs, training=False):
        # A. Sensation
        enc_out = self.backbone.encoder(inputs["token_ids"], padding_mask=inputs["padding_mask"])
        
        # B. Perception (Agent)
        agent_out = self.agent(enc_out)
        
        # Select task (Greedy for now)
        task_idx = tf.argmax(agent_out["route"][0]) 
        
        # C. Modulation
        steered = enc_out + agent_out["steer"]
        plastic = self.lora_proj(steered, adapter_index=task_idx)
        
        # D. Stabilization
        memory = self.vq(plastic)
        
        return memory, agent_out

    def train_step(self, data):
        x, y = data # y would be target text in real scenario
        
        with tf.GradientTape() as tape:
            # Forward
            memory, agent_out = self(x, training=True)
            
            # Mock Reconstruction Loss (Memory should match Input roughly for this test)
            rec_loss = tf.reduce_mean((memory - self.backbone.encoder(x["token_ids"]))**2)
            
            # Aux Losses
            ortho_loss = self.lora_proj.orthorthogonality_loss()
            vq_loss = sum(self.vq.losses)
            
            total_loss = rec_loss + vq_loss + 0.1 * ortho_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        self.ortho_tracker.update_state(ortho_loss)
        return {"loss": total_loss, "ortho": self.ortho_tracker.result()}