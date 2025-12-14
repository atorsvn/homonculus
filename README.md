# LoRA-H²-BART: Hierarchical Homuncular Framework

[![Status](https://img.shields.io/badge/Status-Research_Preview-red)]()
[![Framework](https://img.shields.io/badge/Framework-Keras_3-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

**A Neuro-Symbolic Architecture for Continual Learning and Controlled Generation.**

LoRA-H²-BART is an experimental implementation of a **Self-Regulating Generative Automaton (SRGA)**. It mitigates catastrophic forgetting in Large Language Models by decoupling the learning process into three distinct interacting vector fields: **Plasticity, Stability, and Executive Control.**

> **Read the Agentic Theory:** [agents.md](./agents.md)

---

## 🧠 System Architecture

The system augments a frozen BART backbone with three novel mechanisms:

1.  **Plasticity (Continual LoRA):** Orthogonal low-rank adapters that allow rapid adaptation to new tasks without interfering with prior knowledge.
2.  **Stability (H-RVQ):** A Hierarchical Residual Vector Quantization bottleneck that forces continuous thought into discrete, stable memory anchors.
3.  **Control (The Homunculus):** A "System 2" meta-controller that monitors the encoder and intervenes via steering vectors and routing decisions before generation occurs.

```mermaid
graph TD
    Input[Input Text] --> FrozenEncoder[Frozen Cortex (Encoder)]
    FrozenEncoder -->|Stop Gradient| Homunculus[Homuncular Agent]
    
    Homunculus -->|Routing| LoRA[Select LoRA Adapter]
    Homunculus -->|Steering| Residual[Inject Steering Vector]
    
    FrozenEncoder --> Residual
    Residual --> LoRA
    LoRA --> VQ[Discrete Memory (H-RVQ)]
    VQ --> Decoder[Frozen Decoder]
    Decoder --> Output[Generated Text]
````

-----

## 📦 Installation

This repository requires **Python 3.10+** and **Keras 3**.

```bash
git clone [https://github.com/your-username/lora-h2-bart.git](https://github.com/your-username/lora-h2-bart.git)
cd lora-h2-bart
pip install -r requirements.txt
```

**Requirements:**

  * `keras>=3.0.0`
  * `keras-nlp>=0.8.0`
  * `tensorflow>=2.16.0` (or PyTorch/JAX backend)

-----

## 🚀 Quickstart

Initialize the automaton and perform a forward pass with the Homuncular loop active.

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow" # or 'jax', 'torch'

import tensorflow as tf
from lora_h2_bart.model import LoRA_H2_BART
from lora_h2_bart.config import ModelConfig

# 1. Configure the Automaton
config = ModelConfig(
    preset="bart_base_en",
    hidden_dim=768,
    num_adapters=5,       # Number of distinct tasks/personas
    vq_codebook_size=1024 # Size of discrete memory
)

# 2. Instantiate Model
model = LoRA_H2_BART(config)

# 3. Prepare Sensory Input (Tokenized)
inputs = {
    "token_ids": tf.ones((1, 128), dtype=tf.int32),
    "padding_mask": tf.ones((1, 128), dtype=tf.int32)
}

# 4. Execute Forward Pass
# Returns the quantized memory state and the agent's decisions
memory_state, agent_decisions = model(inputs)

print(f"Active Adapter Index: {tf.argmax(agent_decisions['route'][0])}")
print(f"Steering Vector Magnitude: {tf.norm(agent_decisions['steer'])}")
```

-----

## 🧪 Testing

We use `pytest` for unit and integration testing.

```bash
# Run all tests
pytest tests/

# Run specific integration test
pytest tests/test_integration.py
```

-----

## 📂 Repository Structure

```text
lora-h2-bart/
├── agents.md             # Theoretical specification of the Homuncular Agent
├── lora_h2_bart/
│   ├── config.py         # Hyperparameter dataclasses
│   ├── layers.py         # Custom layers (VQ, LoRA, Controller)
│   └── model.py          # Main Keras Model assembly
└── tests/                # Unit verification
```

## ⚠️ Disclaimer

This is a **research artifact**. It is designed to explore the intersection of Control Theory, Reinforcement Learning, and Transformers. It is not intended for production deployment without significant tuning of the PPO reward signals and VQ codebook utilization.

## 📄 Citation

If you use this architecture in your research, please cite the Technical Report:

> **LoRA-H²-BART: A Hierarchical Homuncular Framework for Continual Learning** \> *Draft Specification v1.0, Dec 2025.*

