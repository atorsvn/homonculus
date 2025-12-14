import keras
import keras_nlp
import tensorflow as tf
from .layers import LoRADense, ResidualVQ, HomuncularController
from .config import ModelConfig

class LoRA_H2_BART_RL(keras.Model):
    """
    LoRA-H²-BART-RL: Active Agent model with Vector DB Router and PPO Action Head.
    """
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # 1. Backbone (Frozen)
        self.backbone = keras_nlp.models.BartBackbone.from_preset(config.preset)
        
        # 2. Plasticity (LoRA bank)
        self.lora_proj = LoRADense(
            keras.layers.Dense(config.hidden_dim),
            rank=config.lora_rank,
            num_adapters=config.num_adapters
        )

        # 3. Stability (VQ)
        self.vq = ResidualVQ(config.num_quantizers, config.vq_codebook_size, config.hidden_dim)
        
        # 4. Agency (PPO Controller)
        self.agent = HomuncularController(config)

        # 5. Vector DB Simulation (Adapter Embeddings)
        self.adapter_embeddings = self.add_weight(
            name="adapter_embeddings",
            shape=(config.num_adapters, config.hidden_dim),
            initializer="uniform",
            trainable=False 
        )
        
        # Metrics
        self.rec_tracker = keras.metrics.Mean(name="rec_loss")
        self.ortho_tracker = keras.metrics.Mean(name="ortho_loss")

    def _vector_db_query(self, query_vector):
        """Simulates querying the Vector DB to find the single best adapter index."""
        # Cosine similarity (L2 normalization is assumed for simplicity)
        normalized_query = tf.nn.l2_normalize(query_vector, axis=-1)
        normalized_embeddings = tf.nn.l2_normalize(self.adapter_embeddings, axis=-1)
        
        # [Num_Adapters, Dim] x [Dim] -> [Num_Adapters]
        similarity = tf.linalg.matvec(normalized_embeddings, normalized_query)
        
        # Get the index of the best match (Batch dimension is 1)
        return tf.argmax(similarity, axis=0, output_type=tf.int32) 

    def call(self, inputs, training=False):
        # A. Sensation (Encoder)
        enc_out = self.backbone.encoder(inputs["token_ids"], padding_mask=inputs["padding_mask"])
        
        # B. Perception (Agent)
        agent_out = self.agent(enc_out)
        
        # C. Vector DB Router (Query uses the mean encoder state from the first element in batch)
        query_vector = tf.reduce_mean(enc_out[0], axis=0)
        task_idx = self._vector_db_query(query_vector)
        
        # D. Modulation (Steering + Plasticity)
        steered = enc_out + agent_out["steer"]
        plastic = self.lora_proj(steered, adapter_index=task_idx)
        
        # E. Stabilization
        memory = self.vq(plastic)
        
        return memory, agent_out, task_idx
    
    # --- PPO Training is handled externally but requires a conceptual function ---
    
    def train_step_rl(self, batch_data):
        """
        Conceptual PPO Update Step (Agent Training). 
        Optimizes the HomuncularController based on collected RPG experience.
        """
        config = self.config
        
        # 1. Prepare data (Requires state, action, old_log_prob, return, advantage from RPG runs)
        
        with tf.GradientTape() as tape:
            # Predict the current policy outputs for the historical states
            agent_outputs = self.agent(batch_data['state'])
            action_logits = agent_outputs['action_logits']
            
            # Policy Loss (PPO Clipping)
            action_mask = tf.one_hot(batch_data['action'], config.rl_action_space_size)
            new_log_probs = tf.reduce_sum(tf.nn.log_softmax(action_logits) * action_mask, axis=1)
            ratio = tf.exp(new_log_probs - batch_data['old_log_probs'])
            
            surrogate_1 = ratio * batch_data['advantage']
            surrogate_2 = tf.clip_by_value(ratio, 1.0 - config.ppo_epsilon, 1.0 + config.ppo_epsilon) * batch_data['advantage']
            policy_loss = -tf.reduce_mean(tf.minimum(surrogate_1, surrogate_2))

            # Value Loss (Critic)
            value_pred = tf.squeeze(agent_outputs['value'])
            value_loss = tf.reduce_mean(tf.square(batch_data['return'] - value_pred))

            # Total PPO Loss
            total_ppo_loss = policy_loss + 0.5 * value_loss
        
        # Apply gradients only to the Agent's variables
        agent_vars = self.agent.trainable_variables
        grads = tape.gradient(total_ppo_loss, agent_vars)
        self.optimizer.apply_gradients(zip(grads, agent_vars))
        
        return {"ppo_loss": total_ppo_loss, "policy_loss": policy_loss, "value_loss": value_loss}


    def train_step(self, data):
        """
        Standard Keras train_step: Used only for updating LoRA/VQ (Skill Acquisition).
        Assumes data (x, y) is provided for a specific task (Adapter k).
        """
        x, y = data
        
        # We use index 0 as placeholder for the active adapter during supervised training
        task_idx = 0 
        
        with tf.GradientTape() as tape:
            # Forward pass (only for the supervised task)
            memory, _, _ = self(x, training=True)
            
            # 1. Reconstruction Loss (Text Generation)
            rec_loss = keras.losses.mean_squared_error(y, memory) 
            
            # 2. Aux Losses
            ortho_loss = self.lora_proj.orthogonality_loss()
            vq_loss = sum(self.vq.losses)
            
            total_loss = rec_loss + self.config.beta_vq * vq_loss + self.config.delta_ortho * ortho_loss

        # Gradients are filtered implicitly by the conditional LoRA forward pass.
        trainable_vars = self.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        
        self.rec_tracker.update_state(rec_loss)
        self.ortho_tracker.update_state(ortho_loss)
        
        return {
            "loss": total_loss,
            "rec_loss": self.rec_tracker.result(),
            "ortho_loss": self.ortho_tracker.result()
        }