import keras
import tensorflow as tf
from .config import ModelConfig

# --- Stability Mechanism ---

class VectorQuantizer(keras.layers.Layer):
    """Discrete bottleneck with Straight-Through Estimator."""
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta 

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.num_embeddings, self.embedding_dim),
            initializer="uniform",
            trainable=True,
        )

    def call(self, x):
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Distances calculation
        similarity = tf.matmul(flattened, self.embeddings, transpose_b=True)
        distances = (
            tf.reduce_sum(flattened**2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings**2, axis=0)
            - 2 * similarity
        )
        
        encoding_indices = tf.argmin(distances, axis=1)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings)
        quantized = tf.reshape(quantized, input_shape)

        # Commitment and Codebook Losses
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-Through Estimator
        return x + tf.stop_gradient(quantized - x)

class ResidualVQ(keras.layers.Layer):
    """Hierarchical VQ wrapper."""
    def __init__(self, num_quantizers, num_embeddings, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.layers = [
            VectorQuantizer(num_embeddings, embedding_dim, name=f"vq_{i}") 
            for i in range(num_quantizers)
        ]

    def call(self, x):
        quantized_out = 0.0
        residual = x
        for layer in self.layers:
            quantized = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
        return quantized_out

# --- Plasticity Mechanism ---

class LoRADense(keras.layers.Layer):
    """Multi-Head LoRA Adapter Layer with Orthogonality Constraint."""
    def __init__(self, original_layer, rank=16, num_adapters=5, **kwargs):
        super().__init__(**kwargs)
        self.original_layer = original_layer
        self.original_layer.trainable = False 
        self.units = original_layer.units
        self.rank = rank
        self.num_adapters = num_adapters
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.lora_A = self.add_weight(
            name="lora_A",
            shape=(self.num_adapters, input_dim, self.rank),
            initializer="he_uniform",
            trainable=True
        )
        self.lora_B = self.add_weight(
            name="lora_B",
            shape=(self.num_adapters, self.rank, self.units),
            initializer="zeros",
            trainable=True
        )

    def call(self, inputs, adapter_index=0):
        frozen_out = self.original_layer(inputs)
        
        # Select the active adapter based on index (chosen by Vector DB/Router)
        A = tf.gather(self.lora_A, adapter_index) 
        B = tf.gather(self.lora_B, adapter_index)
        
        lora_out = tf.matmul(inputs, A)
        lora_out = tf.matmul(lora_out, B)
        return frozen_out + (lora_out * (1.0 / self.rank))

    def orthogonality_loss(self):
        """Calculates orthogonality between different adapter matrices."""
        # This is added to the total loss to ensure task separation.
        flat_A = tf.reshape(self.lora_A, (self.num_adapters, -1))
        gram = tf.matmul(flat_A, flat_A, transpose_b=True)
        identity = tf.eye(self.num_adapters)
        # We penalize all off-diagonal interactions
        return tf.reduce_mean((gram * (1 - identity)) ** 2)

# --- Homuncular RL Agent ---

class HomuncularController(keras.layers.Layer):
    """
    The Active RL Agent: Manages resource selection and outputs the next game command.
    """
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = config.hidden_dim
        self.action_space_size = config.rl_action_space_size 

        # Policy Network (processes the state)
        self.policy_net = keras.Sequential([
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dense(128, activation="relu")
        ])
        
        # HEAD 1: Steering Head (Continuous action for tone/style)
        self.steer = keras.layers.Dense(self.hidden_dim, activation="tanh", name="steer")
        
        # HEAD 2: VALUE Head (Estimates future reward for PPO)
        self.value_head = keras.layers.Dense(1, name="value")

        # HEAD 3: ACTION Head (Discrete action for game play)
        self.action_head = keras.layers.Dense(self.action_space_size, activation=None, name="action_logits")

    def call(self, inputs):
        # Stop gradient is CRITICAL to keep the agent policy separate
        x = tf.stop_gradient(inputs)
        context = tf.reduce_mean(x, axis=1) # [B, Seq, Dim] -> [B, Dim]
        
        features = self.policy_net(context)
        
        return {
            "steer": tf.expand_dims(self.steer(features), 1), 
            "value": self.value_head(features),
            "action_logits": self.action_head(features)
        }