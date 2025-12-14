from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Backbone
    preset: str = "bart_base_en"
    hidden_dim: int = 768
    
    # Plasticity (LoRA)
    lora_rank: int = 16
    num_adapters: int = 256 # Scaled up for Vector DB
    
    # Stability (VQ)
    vq_codebook_size: int = 1024
    vq_embedding_dim: int = 768
    num_quantizers: int = 4
    
    # Reinforcement Learning (New parameters for RPG agent)
    rl_action_space_size: int = 8 # E.g., NORTH, SOUTH, ATTACK, PICKUP
    ppo_epsilon: float = 0.2
    
    # Weighting Factors (Simplified for training)
    beta_vq: float = 0.01
    delta_ortho: float = 0.1