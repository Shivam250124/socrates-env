# SOCRATES Demo Script (3-5 Minutes)

## Setup

**Materials Needed**:
- Laptop with slides/terminal
- results/ folder with generated artifacts
- Server running locally or on HF Spaces

**Timing**: 3-5 minutes total

---

## [30 seconds] The Hook

> "LLMs are extraordinary at answering questions. But the best teachers in history — Socrates, Feynman, Richard Hamming — almost never gave direct answers. They asked questions. We built an RL environment to teach this exact skill."

**Visual**: Show title slide with Socrates bust

---

## [60 seconds] The Baseline (The Bad)

> "Let me show you what happens when you ask any LLM to be a Socratic tutor."

**Visual**: Show `example_baseline_dialogue.txt`

**Read aloud** (abbreviated):
```
Student: Why does 0.1 + 0.2 != 0.3 in Python?
Baseline Model: Can you tell me more about that?
Student: Well, they're such simple numbers...
Baseline Model: Why do you think that is?
Student: I don't know, maybe it's a bug?
[continues with generic questions]
```

> "It asks generic questions. No progress. The student is still confused after 8 steps. This is what every LLM does — it cannot resist eventually explaining the answer."

**Visual**: Show baseline metrics:
- Success rate: 18%
- Direct answer rate: 64%
- Mean reward: 0.30

---

## [60 seconds] The Environment

> "So we built SOCRATES — an RL environment specifically designed to train this capability."

**Visual**: Show architecture diagram (from README)

**Key points** (rapid fire):
- **Simulated student** with defined misconceptions (deterministic state machine)
- **8 programming concepts** (floating point, recursion, mutable defaults, etc.)
- **5 reward signals** that distinguish good questions from bad ones:
  1. Teaching Progress (40%) — Did understanding increase?
  2. Socratic Compliance (25%) — Did you reveal the answer? (penalty)
  3. Question Quality (15%) — Open-ended vs yes/no
  4. Efficiency (10%) — Fewer steps bonus
  5. Misconception Targeting (10%) — Hitting the right weak spot

> "The critical innovation: efficiency is gated by compliance. If you cheat by giving the answer, you get no speed bonus."

---

## [60 seconds] The Trained Model (The Good)

> "After 500 episodes of training, here's what the model learned."

**Visual**: Show `example_trained_dialogue.txt`

**Read aloud** (abbreviated):
```
Student: Why does 0.1 + 0.2 != 0.3 in Python?
Trained Model: How do you think computers store numbers internally?
Student: Hmm, something different than regular numbers?
Trained Model: Can you think of a number in base 10 that repeats forever?
Student: Like 1/3? That's 0.333 forever...
Trained Model: Exactly! Now what happens when a computer tries to store 0.1 in binary?
Student: Oh! So 0.1 repeats in binary too?
[Student arrives at understanding in 6 steps]
```

> "The model learned to never give the answer. It asks questions that force the student to reason. The student arrives at the understanding themselves. That is Socratic teaching."

**Visual**: Show trained metrics:
- Success rate: 67% (3.7× improvement)
- Direct answer rate: 9% (85% reduction)
- Mean reward: 0.75 (150% increase)

---

## [30 seconds] The Curves

**Visual**: Show `reward_curves.png`

> "Here's the learning curve. Baseline is flat — generic questions don't work. Training shows clear upward trend. The model is learning to ask better questions."

**Visual**: Show `success_rate.png`

> "Success rate goes from 18% to 67%. The model genuinely learned a new capability."

---

## [30 seconds] The Stakes

> "Why does this matter?"

**Key points**:
- **300 million students globally** have no access to quality tutoring
- Research shows AI tutors that **question** are more effective than AI tutors that **explain**
- This is one of the most impactful applications of aligned AI

> "More broadly, this demonstrates that RL can train AI systems for restraint, not just capability. The hardest part of Socratic teaching isn't knowing the answer — it's having the discipline to not say it."

---

## [Optional: 30 seconds] Live Demo

**If time permits and server is running**:

```bash
# Terminal demo
python -c "
from client import SocratesEnv
from models import SocratesAction

with SocratesEnv() as env:
    obs = env.reset('foundation')
    print(f'Student: {obs.student_response}')
    
    obs, reward, done, info = env.step(
        SocratesAction(question='What do you think happens when you try that?')
    )
    print(f'Reward: {reward:.3f}')
    print(f'Student: {obs.student_response}')
"
```

> "The environment is live, open source, and ready to use."

---

## Closing

> "SOCRATES is the first RL environment for Socratic teaching. Zero prior art. Novel capability. Paper-worthy methodology. And it works."

**Visual**: Show final slide with:
- GitHub link
- HuggingFace Space link
- Contact info

> "Thank you. Questions?"

---

## Backup Slides (If Asked)

### Technical Details
- **Model**: Qwen-2.5-1.5B with LoRA
- **Training**: GRPO (Group Relative Policy Optimization)
- **Efficiency**: Unsloth for 2× speedup
- **Episodes**: 500 (4-6 hours on single GPU)
- **Curriculum**: 3 phases (foundation → intermediate → advanced)

### Anti-Hacking Measures
1. Answer keyword detection (2+ keywords = -1.5 penalty)
2. Rhetorical pattern detection ("So wouldn't it be...")
3. Leading question detection ("Isn't it because...")
4. Repeat question penalty (embedding similarity > 0.85)
5. Hard rules (min/max length, must have "?")
6. Efficiency × compliance coupling

### Concept Bank
- Floating point representation
- Recursive termination
- Mutable default arguments
- Zero-based indexing
- Boolean operator return values
- Integer division
- Pass by reference
- Negative modulo

### Future Work
- Expand to 50+ concepts
- Multi-turn reasoning chains
- Adaptive difficulty per student
- Real student evaluation
- Integration with Khan Academy / Coursera

---

## Q&A Preparation

**Q: How do you know the student actually learned?**  
A: The student is a deterministic state machine with an internal understanding_level scalar (0.0 to 1.0). We track this throughout the episode. Success = understanding >= 0.85.

**Q: Isn't this just prompt engineering?**  
A: No. We tried extensive prompt engineering — it fails within 2-3 turns. The model needs RL to learn the delayed gratification of Socratic teaching.

**Q: What if the model just memorizes good questions?**  
A: The reward is tied to understanding delta, not question similarity. Memorized questions that don't advance understanding get low reward.

**Q: How do you prevent reward hacking?**  
A: 5 independent reward signals + 6 anti-hacking measures. Most importantly: efficiency is gated by compliance. Fast success via cheating gets no bonus.

**Q: Can this work with real students?**  
A: Yes! The deterministic student is for training. Once trained, the model can interact with real students. We're planning human evaluation next.

**Q: What's the compute cost?**  
A: 500 episodes on a single A100 = 4-6 hours. Total cost: ~$20-30 on cloud GPUs.

**Q: Why not use a larger model?**  
A: We wanted fast iteration for the hackathon. The 1.5B model proves the concept. Scaling to 7B+ is straightforward.

**Q: Is this publishable?**  
A: Absolutely. "Reinforcement Learning for Socratic Pedagogical Agents" — AIED, NeurIPS Education Workshop, or EMNLP. Zero prior art in this domain.
