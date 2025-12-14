from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Backbone
    preset: str = "bart_base_en"
    hidden_dim: int = 768
    
    # Plasticity (LoRA)
    lora_rank: int = 16
    num_adapters: int = 5
    
    # Stability (VQ)
    vq_codebook_size: int = 1024
    vq_embedding_dim: int = 768
    num_quantizers: int = 4
    
    # Executive
    use_pondering: bool = True