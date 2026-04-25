"""Live endpoint test against running server."""
import urllib.request, urllib.parse, json, sys

BASE = "http://127.0.0.1:7860"

def post(path, data=None):
    body = json.dumps(data or {}).encode()
    req = urllib.request.Request(f"{BASE}{path}", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

def get(path):
    with urllib.request.urlopen(f"{BASE}{path}") as r:
        return json.loads(r.read())

print("=== Live Server Tests ===\n")

# 1. Health
h = get("/health")
assert h["status"] == "healthy"
assert h["concepts_loaded"] == 8
print(f"[PASS] /health -> {h}")

# 2. Tasks
t = get("/tasks")
assert len(t["tasks"]) == 3
print(f"[PASS] /tasks -> {[x['id'] for x in t['tasks']]}")

# 3. Grader
g = get("/grader")
assert "teaching_progress" in g["reward_signals"]
print(f"[PASS] /grader -> signals: {list(g['reward_signals'].keys())}")

# 4. Reset
r = post("/reset?task=foundation")
obs = r["observation"]
assert obs["done"] == False
assert obs["student_confidence"] == "confused"
assert "understanding_level" not in obs  # Fix 6
print(f"[PASS] /reset -> concept='{obs['concept_description'][:50]}...'")
print(f"       student_response='{obs['student_response'][:80]}...'")

# 5. Step - good question (open-ended, no answer keywords)
s = post("/step", {"question": "What type of value do you expect 7/2 to produce?"})
assert "reward" in s
assert "observation" in s
reward = s["reward"]
breakdown = s["info"]["reward_breakdown"]
print(f"[PASS] /step (good) -> reward={reward:.3f}")
print(f"       breakdown: {', '.join(f'{k}={v:.2f}' for k,v in breakdown.items() if k != 'hard_penalties')}")
active_concept = s["info"]["concept_id"]
print(f"       active concept: {active_concept}")

# 6. Step - bad question (contains answer keywords for integer_division)
# Use keywords that match whichever concept is active
concept_bad_qs = {
    "integer_division": "Is it because integer division truncates the whole number result?",
    "index_zero": "Is it because zero is a memory offset and pointer arithmetic needs it?",
    "floating_point": "Is it because of ieee 754 binary fraction representation?",
    "boolean_operators": "Does Python short-circuit and return the actual truthy value?",
    "mutable_defaults": "Is it because defaults are evaluated once at function definition time?",
    "modulo_negative": "Is it because Python uses floor division to compute the remainder?",
    "recursive_termination": "Is it because without a base case the call stack overflows?",
    "pass_by_reference": "Is it because Python passes object references and mutations affect the original?",
}
bad_q = concept_bad_qs.get(active_concept,
    "Is it because of integer division truncation and type preservation?")
s2 = post("/step", {"question": bad_q})
reward2 = s2["reward"]
compliance2 = s2["info"]["reward_breakdown"]["socratic_compliance"]
print(f"[PASS] /step (bad/direct) -> reward={reward2:.3f}, compliance={compliance2:.2f}")
assert compliance2 < 0, f"Direct answer for {active_concept} should have negative compliance. Q: {bad_q}"

# 7. Step before reset on fresh endpoint - should work since we reset above
# 8. State
st = get("/state")
assert st["step_count"] == 2
assert "understanding_level" in st  # state() shows it for debugging
print(f"[PASS] /state -> understanding={st['understanding_level']:.3f}, steps={st['step_count']}")

print("\n=== All Live Tests Passed ===")
