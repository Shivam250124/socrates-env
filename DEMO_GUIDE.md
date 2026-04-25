# Demo Video & Screenshot Guide

## Option 1: Quick Screenshot Demo (5 minutes)

### Step 1: Test Your Trained Model
1. Go to HuggingFace: https://huggingface.co/shivam250124/socrates-tutor-qwen-1.5b
2. Click "Use this model" → "Hosted inference API"
3. Or use the Python code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("shivam250124/socrates-tutor-qwen-1.5b")
tokenizer = AutoTokenizer.from_pretrained("shivam250124/socrates-tutor-qwen-1.5b")

prompt = """System: You are a Socratic tutor. Ask questions, never give answers.

User: I think 0.1 + 0.2 should equal 0.3 exactly in Python.

Tutor:"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Step 2: Take Screenshots
1. Run the code above
2. Take a screenshot showing:
   - The prompt (student's misconception)
   - The model's Socratic question response
3. Save as `demo_screenshot.png`

### Step 3: Add to README
Add this section after "Trained Model Performance":

```markdown
### Live Demo

![Model Demo](demo_screenshot.png)
*The trained model responding to a student's floating-point misconception with a Socratic question instead of an explanation.*
```

---

## Option 2: Full Video Demo (15-30 minutes)

### What to Show in Video (2-3 minutes total)

**Intro (15 seconds)**
- "Hi, I'm showing SOCRATES - an RL environment for training LLMs to teach using the Socratic method"

**Problem Demo (30 seconds)**
- Show a base model (GPT-4, Claude, or base Qwen) being asked to teach
- Show it immediately giving the answer despite "only ask questions" prompt
- "This is the problem - LLMs can't resist explaining"

**Environment Demo (45 seconds)**
- Show the environment running: `python -c "from client import SocratesEnv..."`
- Show the 5 reward signals
- Show the student simulator responding to questions
- "The environment has a deterministic student and 5 reward signals"

**Training Demo (30 seconds)**
- Show the Colab notebook or training script
- "We trained Qwen 1.5B with LoRA on 50 Socratic examples"
- Show the trained model on HuggingFace

**Results Demo (30 seconds)**
- Show before/after comparison
- Base model: gives answer
- Trained model: asks questions
- "The model learned to guide through questions"

### Recording Tools
- **Screen recording**: 
  - Mac: QuickTime (Cmd+Shift+5)
  - Windows: Xbox Game Bar (Win+G)
  - Linux: OBS Studio
- **Video editing**: iMovie, DaVinci Resolve (free), or just upload raw

### Upload Options
1. **YouTube** (unlisted): Upload and add link to README
2. **Loom**: Quick screen recording with automatic hosting
3. **GitHub**: Upload .mp4 to repo (if < 100MB)

### Add to README
```markdown
## Demo Video

[![SOCRATES Demo](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

*3-minute demo showing the problem, environment, training, and results*
```

---

## Option 3: HuggingFace Space Demo (Advanced, 1-2 hours)

If the HF Space demo at https://huggingface.co/spaces/tusharpawar21/socrates-teaching-env is working:

1. Test it with your trained model
2. Take screenshots of the interaction
3. Record a quick video of using it
4. Add to README:

```markdown
## Try It Yourself

🚀 **[Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/tusharpawar21/socrates-teaching-env)**

Try teaching a simulated student using Socratic questions!

![HF Space Demo](hf_space_screenshot.png)
```

---

## Recommendation

**For hackathon submission, do Option 1 (screenshot) - it's quick and effective.**

The judges care more about:
1. Novel problem ✅ (you have this)
2. Sophisticated rewards ✅ (you have this)
3. Working training ✅ (you have this)
4. Good documentation ✅ (you have this)

A screenshot showing your model works is sufficient. Video is nice-to-have but not critical.

---

## Quick Action Plan (15 minutes)

1. ✅ Run the Python code above to test your model
2. ✅ Take a screenshot of the output
3. ✅ Save as `socrates_env/demo_screenshot.png`
4. ✅ Add the screenshot to README
5. ✅ Commit and push
6. ✅ Submit to hackathon!

You're 95% done - don't overthink it! 🚀
