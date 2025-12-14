# Agentic Specification: The Homuncular Meta-Controller

## 1. Ontology of the Agent
In the LoRA-H²-BART architecture, "Agency" is not an emergent property of the text generation, but an explicit architectural component known as the **Homuncular Meta-Controller (HMC)**.

Unlike standard LLMs which are "reflexive" (Input $\rightarrow$ Output), this system is "reflective." The HMC acts as a **System 2** observer that monitors the **System 1** (Frozen BART Backbone) thought process and intervenes before action (generation) is taken.

## 2. The Agent's Action Space
The HMC ($\pi_\phi$) does not generate tokens. It manipulates the *conditions* of generation.

| Action Type | Mechanism | Cognitive Analogue |
| :--- | :--- | :--- |
| **Routing** | `adapter_idx = argmax(policy(h))` | **Context Switching**: Deciding *which* personality or skill set is required for the current task (e.g., Coding vs. Poetry). |
| **Steering** | `h_mod = h + v_steer` | **Bias Injection**: Deliberately shifting emotional tone or focus (e.g., "Be more polite" or "Focus on facts"). |
| **Pondering** | `while ponder_gate > 0.5: refine(h)` | **Rumination**: Looping on a thought to increase clarity before speaking. |

## 3. Cognitive Control Loop
The agent operates within the `Perception-Modulation-Action` loop:

1.  **Sensation (Input):** The frozen encoder processes raw text. The Agent receives a `stop_gradient` copy of this state. This ensures the Agent cannot "hallucinate" or alter the raw sensory data to suit its preferences; it must deal with reality as it is.
2.  **Deliberation (Policy):** The Agent computes a policy distribution over the available LoRA adapters and calculates a continuous steering vector.
3.  **Intervention (Modulation):** The Agent physically alters the neural activations of the backbone model, injecting its intent into the residual stream.
4.  **Commitment (Quantization):** The modified thought is "collapsed" into a discrete code from the VQ Codebook. This is the moment a fluid thought becomes a concrete memory.

## 4. Training the Agent
The Agent is not trained via next-token prediction. It is trained via **Reinforcement Learning (PPO)** or explicit supervision to maximize high-level objectives (e.g., Safety, Orthogonality, Task Success), distinct from the language model's objective (Perplexity).