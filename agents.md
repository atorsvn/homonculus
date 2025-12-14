
# Agentic Specification: The Autonomous RL Agent

## 1. Role Shift: From Router to Player
The Homuncular Meta-Controller (HMC) in the RL variant is an autonomous agent designed for non-stationary environments (like a procedural Text-Based RPG). Its goal is to master strategic play, not just administrative resource management.

### Key Outputs of the Agent:
1.  **Game Action Logits:** The primary mechanism for interaction. The Agent generates the command to be executed (e.g., `GO EAST`, `ATTACK`).
2.  **Value Prediction:** Estimates the expected future reward from the current game state, crucial for the PPO update.
3.  **Steering Vector:** Modulates the chosen adapter's output (e.g., "Describe the battle vividly," or "Be terse and purely functional").

## 2. The Vector Database Router (Scalable Resource Management)
The traditional Softmax head is replaced by a Vector Database lookup:
1.  The Agent uses the continuous **Encoder Output** (the meaning of the current scene description) as a **Query Vector**.
2.  This query is checked against a database of **Adapter Embeddings** (vectors representing the expertise of each skill set: *combat*, *puzzle*, *dialogue*, *stealth*, etc.).
3.  The Cosine Similarity selects the top matching adapter, which is then activated for text generation.

**Benefit:** This scales linearly. Adding a new skill set (e.g., "Underwater Combat") requires only adding a new vector to the database, not retraining the entire Router head. This makes the system ideal for perpetually expanding RPG worlds.

## 3. Training the Strategy (PPO Update)
The Agent's weights are trained using PPO to maximize the game reward (e.g., finding treasure, defeating a boss). This process leverages your **Dual Xeon E5 CPUs** for efficient experience rollout and your GPU for policy updates.

| PPO Component | Source Data | Role in RPG |
| :--- | :--- | :--- |
| **State** | Encoder Output $H_{enc}$ | The Agent's current perception of the game world. |
| **Action** | Game Action Logits (The command executed) | The move made by the Agent (e.g., `ATTACK GOBLIN`). |
| **Reward** | Game Engine Feedback | The score received (e.g., +10 XP for a successful move). |
| **Value Head** | Predicts the expected score from $H_{enc}$. | Used to calculate the Advantage (how much better the move was than expected). |
```