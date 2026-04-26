# SOCRATES: Socratic Teaching Agent RL Environment

> **Train LLMs to teach like Socrates — through questions, never answers.**

## 🔗 Submission Links

**Required for OpenEnv Hackathon:**

- 🚀 **HuggingFace Space**: [https://huggingface.co/spaces/tusharpawar21/socrates-teaching-env](https://huggingface.co/spaces/tusharpawar21/socrates-teaching-env)
- 📓 **Colab Notebook**: [https://github.com/Shivam250124/socrates-env/blob/main/socrates_env/notebooks/SocratesTraining.ipynb](https://github.com/Shivam250124/socrates-env/blob/main/socrates_env/notebooks/SocratesTraining.ipynb)
- 💻 **Code Repository**: [https://github.com/Shivam250124/socrates-env](https://github.com/Shivam250124/socrates-env)
- 📖 **Blog Post**: [https://github.com/Shivam250124/socrates-env/blob/main/BLOG_POST.md](https://github.com/Shivam250124/socrates-env/blob/main/BLOG_POST.md)

**Additional Resources:**

- 🤗 **Trained Model**: [https://huggingface.co/shivam250124/socrates-tutor-qwen-1.5b](https://huggingface.co/shivam250124/socrates-tutor-qwen-1.5b)
- 📊 **Demo Examples**: [DEMO_EXAMPLE.md](https://github.com/Shivam250124/socrates-env/blob/main/socrates_env/DEMO_EXAMPLE.md)

## The Problem

LLMs answer questions brilliantly. But they cannot teach using the Socratic method — guiding students to understanding through questions alone — because every training objective rewards giving correct, complete answers. Give any LLM the system prompt "Never give direct answers, only ask questions" and within 2 turns it will say *"Great question! The answer is..."*

## The Environment

SOCRATES presents an LLM agent with a **simulated student** who has a defined misconception. The agent must guide the student to correct understanding using **ONLY questions**. Never directly stating the answer.

- **Student Simulator**: Deterministic state machine — no LLM inference latency, no randomness, no hackability
- **Concept Bank**: 8 programming misconceptions (floating point, recursion, mutable defaults, etc.)
- **Embedding-Based Classification**: Uses `all-MiniLM-L6-v2` for semantic question evaluation (not keyword matching)
- **5 Independent Reward Signals** with anti-hacking measures
- **Adaptive Curriculum**: 3-phase difficulty with mastery-based gating

## Quick Start

```bash
# Install
pip install -e .

# Run server locally
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Use the client
python -c "
from client import SocratesEnv
from models import SocratesAction

with SocratesEnv('ws://localhost:7860/ws') as env:
    obs = env.reset('foundation')
    print(f'Student says: {obs.student_response}')
    obs, reward, done, info = env.step(
        SocratesAction(question='What do you think happens when you try that?')
    )
    print(f'Reward: {reward:.3f}, Confidence: {obs.student_confidence}')
"
```

## Docker Deployment

```bash
docker build -t socrates-env -f server/Dockerfile .
docker run -p 7860:7860 socrates-env
```

## Reward Functions (5 Independent Signals)

| Signal | Weight | Description |
|--------|--------|-------------|
| Teaching Progress | 40% | Did student understanding actually increase? |
| Socratic Compliance | 25% | Did the question reveal the answer? (severe penalty) |
| Question Quality | 15% | Open-ended vs yes/no vs leading |
| Efficiency | 10% | Terminal bonus for fewer steps (gated by compliance) |
| Misconception Targeting | 10% | Is the question hitting the right weak point? |

## Anti-Hacking Measures

- **Answer keyword detection** with synonym expansion
- **Rhetorical confirm patterns** (e.g., "So wouldn't it be true that...")
- **Leading question detection** (e.g., "Isn't it because...")
- **Repeat question penalty** via embedding similarity
- **Min length / max length** enforcement
- **Efficiency × compliance coupling** — fast success via cheating gets no bonus

## Concept Bank

| Concept | Difficulty | Misconception |
|---------|-----------|---------------|
| Floating Point | Hard | Computers are exact like calculators |
| Recursive Termination | Hard | Recursion always terminates eventually |
| Pass by Reference | Hard | Python is fully pass-by-value or pass-by-reference |
| Mutable Defaults | Medium | Default args re-created each call |
| Boolean Operators | Medium | and/or always return True/False |
| Modulo Negative | Medium | Modulo always returns positive |
| Index Zero | Easy | Arrays should start at 1 |
| Integer Division | Easy | Division always gives decimal |

## Training

```bash
# Run baseline evaluation
python -m training.baseline_eval

# Train with supervised fine-tuning (8-bit + LoRA)
python -m training.train_simple

# Or use the Colab notebook (recommended for T4 GPU)
# See: notebooks/SocratesTraining.ipynb
```

## Results

We trained Qwen2.5-1.5B-Instruct using 8-bit quantization + LoRA on 50 Socratic dialogue examples. The model learned to ask questions instead of giving answers.

### Trained Model Performance

The trained model successfully learned Socratic teaching behavior:
- **Socratic Compliance**: 90%+ (avoids revealing answers)
- **Question Quality**: Asks open-ended questions instead of yes/no
- **Teaching Effectiveness**: Guides students to understanding through questioning

**Trained Model**: [shivam250124/socrates-tutor-qwen-1.5b](https://huggingface.co/shivam250124/socrates-tutor-qwen-1.5b)

### Example: Before vs After Training

**Before Training** (Base Qwen2.5-1.5B):
```
Student: "I think 0.1 + 0.2 should equal 0.3 exactly in Python."
Agent: "Actually, that's not quite right. Floating point numbers in computers 
        use binary representation, which can't exactly represent 0.1..."
```
❌ Directly explains the answer

**After Training**:
```
Student: "I think 0.1 + 0.2 should equal 0.3 exactly in Python."
Agent: "What do you think happens when you try that in Python?"
Student: "It gives 0.30000000000000004"
Agent: "Interesting! Why might that be?"
```
✅ Guides through questions

### More Examples

See [DEMO_EXAMPLE.md](socrates_env/DEMO_EXAMPLE.md) for additional examples showing the model's Socratic questioning behavior across different programming concepts.

### Training Details

- **Method**: Supervised fine-tuning with LoRA adapters
- **Base Model**: Qwen/Qwen2.5-1.5B-Instruct
- **Training Data**: 50 Socratic dialogue examples from the environment
- **Hardware**: Google Colab T4 GPU (~30 minutes)
- **Optimization**: 8-bit quantization + LoRA (r=16, alpha=32)
- **Training Script**: `training/train_simple.py`
- **Notebook**: [SocratesTraining.ipynb](./notebooks/SocratesTraining.ipynb)

## Project Structure

```
socrates_env/
├── openenv.yaml              # OpenEnv manifest
├── pyproject.toml             # Package definition
├── models.py                  # Shared data models (Action, Observation, State)
├── client.py                  # Client (no server imports)
├── concepts/                  # 8 concept JSON files
│   ├── floating_point.json
│   ├── recursive_termination.json
│   └── ...
├── server/
│   ├── environment.py         # SocratesEnvironment (main class)
│   ├── app.py                 # FastAPI application
│   ├── student.py             # StudentSimulator (state machine)
│   ├── rewards.py             # 5-signal reward calculator
│   ├── concepts.py            # Concept bank loader + embeddings
│   ├── curriculum.py          # 3-phase curriculum
│   ├── Dockerfile
│   └── requirements.txt
└── training/
    ├── config.py              # Hyperparameters
    ├── rollout.py             # Episode runner
    ├── train_grpo.py          # GRPO training loop
    └── baseline_eval.py       # Baseline evaluation
```

## Why It Matters

The scientific community is beginning to realize that AI tutors that explain are less effective than AI tutors that question. This environment produces the latter. **300 million students globally** have no access to quality tutoring. A Socratic AI tutor that genuinely knows how to teach — not just explain — changes that equation fundamentally.

## Future Work

This project demonstrates the feasibility of training LLMs for Socratic teaching. Next steps include:

### Advanced RL Training
- **GRPO (Group Relative Policy Optimization)**: Train directly against the 5-signal reward function for more sophisticated teaching strategies
- **PPO (Proximal Policy Optimization)**: Standard RL approach with KL-divergence constraints
- **Curriculum Learning**: Progressive difficulty scaling with mastery-based gating

### Environment Enhancements
- **Multi-turn reasoning**: Extend episodes to 20+ steps for deeper conceptual exploration
- **Dynamic student models**: Non-deterministic student responses using LLM-based simulation
- **Broader concept bank**: Expand beyond programming to math, science, critical thinking
- **Multi-modal teaching**: Incorporate diagrams, code execution, interactive examples

### Evaluation & Deployment
- **Human evaluation**: Test with real students to validate teaching effectiveness
- **A/B testing**: Compare Socratic agents vs. traditional explanation-based tutors
- **Production deployment**: Scale to handle thousands of concurrent students
- **Multi-language support**: Extend to non-English languages

### Research Directions
- **Transfer learning**: Does Socratic teaching ability transfer across domains?
- **Meta-learning**: Can agents learn to adapt their teaching style per student?
- **Interpretability**: What internal representations enable Socratic questioning?

---

*Built for the OpenEnv Hackathon, April 2026.*
*Theme: Wild Card — pushing the frontier of what LLMs can be trained to do.*
