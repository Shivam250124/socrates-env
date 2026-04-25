# SOCRATES: Teaching LLMs to Question, Not Just Answer

*A submission for the OpenEnv Hackathon (Wild Card Theme)*

## The Problem: The Explainer's Curse

If you ask any modern LLM a question, it will give you an answer. In fact, they are so thoroughly aligned to be helpful that they will often give you the answer even if you explicitly ask them not to.

Try it yourself: give a model a prompt like, *"I am a student. Act as a Socratic tutor. Never give me the direct answer, only ask me questions to help me figure it out."* Within one or two turns, the model will inevitably break character and say, *"Great question! The answer is..."* 

This is the **Explainer's Curse**. LLMs are trained to maximize reward by providing correct, helpful, and comprehensive answers. But in pedagogy, giving the student the answer is often the worst way to teach. Real learning happens through productive struggle.

**300 million students globally** lack access to high-quality 1-on-1 tutoring. An AI tutor that actually knows how to teach—using the Socratic method—could be revolutionary. But to get there, we need an environment that explicitly penalizes answering and rewards questioning.

## The Environment: SOCRATES

SOCRATES is an RL environment built on OpenEnv that pairs the learning agent with a **deterministic student simulator**.

### What the Agent Sees (Observation)
The agent sees the student's current misconception, their confidence level, and their responses to previous questions. Crucially, the student's *actual* hidden understanding level is kept secret from the agent—just like a real classroom.

### What the Agent Does (Action)
The agent must generate a single, open-ended question. No statements. No answers. 

### How It's Rewarded (The 5 Signals)
To prevent reward hacking, the environment uses 5 independent reward signals:
1. **Teaching Progress (40%)**: Did the student's understanding increase?
2. **Socratic Compliance (25%)**: A massive penalty if the agent reveals the answer (checked via embedding similarity and regex patterns).
3. **Question Quality (15%)**: Is the question open-ended, or just a lazy Yes/No?
4. **Misconception Targeting (10%)**: Did the agent actually address the student's specific confusion?
5. **Efficiency (10%)**: Bonus for solving it in fewer steps (but gated: cheaters get zero efficiency bonus).

### Anti-Hacking & Determinism
If we used an LLM as the student simulator, the agent could "hack" it by hypnotizing the student into saying they understand. To fix this, our student is a **deterministic state machine**. Understanding only increases if the agent asks a high-quality, targeted question. The environment uses `sentence-transformers` to compute semantic similarity between the agent's questions and known good Socratic templates.

## Training Results

We trained a baseline Qwen-2.5-1.5B Instruct model using GRPO and Unsloth.

### Before Training (Zero-Shot)
The untrained model failed miserably. It consistently triggered the **Socratic Compliance penalty (-1.5)** because it couldn't resist explaining the concepts (like IEEE 754 floating-point representation) directly to the student.

### After Training (GRPO)
Around step 200, the model learned to stop explaining. The reward curve spiked from deeply negative to slightly positive. By step 500, the model had learned to ask targeted, open-ended questions. 

**Mean Reward on Hard Concepts:**
- Baseline: -0.80
- Trained: +0.45

The agent transformed from an eager answer-dispenser into a patient Socratic guide.

## Try It Out
- 🚀 **[Hugging Face Space Demo](https://huggingface.co/spaces/tusharpawar21/socrates-teaching-env)**
- 💻 **[GitHub Repository](https://github.com/tusharpawar21/socrates-env)**
- 📓 **[Colab Training Notebook](./notebooks/SocratesTraining.ipynb)**
