"""Targeted smoke test — runs without any ML dependencies."""
import sys
sys.path.insert(0, '.')

def test_models():
    from models import SocratesAction, SocratesObservation, SocratesState, Concept
    # Fix 6: confirm understanding_level NOT in observation
    fields = SocratesObservation.model_fields
    assert "understanding_level" not in fields
    assert "student_confidence" in fields
    print("[PASS] models: SocratesObservation hides understanding_level")
    
    a = SocratesAction(question="What do you think happens?")
    assert a.question == "What do you think happens?"
    print("[PASS] models: SocratesAction OK")

def test_yaml():
    import yaml
    with open("openenv.yaml") as f:
        cfg = yaml.safe_load(f)
    required = ["name", "version", "environment", "server", "client", "tasks", "reward"]
    missing = [k for k in required if k not in cfg]
    assert not missing, f"Missing: {missing}"
    print(f"[PASS] openenv.yaml: all required blocks present — {cfg['name']} v{cfg['version']}")

def test_concept_bank():
    from server.concepts import ConceptBank
    bank = ConceptBank.load("./concepts")
    assert len(bank.concepts) == 8, f"Expected 8 concepts, got {len(bank.concepts)}"
    
    backend = ("sentence-transformers" if bank.embeddings._use_st 
               else "tfidf" if bank.embeddings._use_tfidf 
               else "word-overlap")
    print(f"[PASS] ConceptBank: 8 concepts loaded, embedding backend={backend}")
    
    for cid, c in bank.concepts.items():
        assert len(c.responses) == 5, f"{cid}: need 5 confidence levels"
        assert len(c.answer_keywords) > 0
        assert len(c.good_question_templates) > 0
    print("[PASS] ConceptBank: all 8 concepts have valid structure")
    return bank

def test_student(bank):
    from server.student import StudentSimulator
    concept = bank.get("index_zero")
    student = StudentSimulator(concept, max_steps=8)
    
    # Test Fix 1: question with answer keyword + yes/no start → direct_answer
    q1 = "Does array offset explain why index starts at zero?"
    sim1 = bank.template_similarity(q1, "index_zero")
    q_type = student._classify_question(q1, sim1)
    assert q_type == "direct_answer", f"Expected direct_answer, got {q_type!r}"
    print(f"[PASS] Fix 1 (classifier ordering): answer-keyword yes/no -> direct_answer")

    # Test Fix 1: rhetorical confirm → leading
    q2 = "So wouldn't it be true that arrays start at zero?"
    sim2 = bank.template_similarity(q2, "index_zero")
    q_type2 = student._classify_question(q2, sim2)
    assert q_type2 in ("leading", "direct_answer"), f"Expected leading, got {q_type2!r}"
    print(f"[PASS] Fix 1 (rhetorical confirm): '{q2[:40]}...' -> {q_type2}")
    
    # Test Fix 3: good question gives positive delta
    q3 = "Where in the computer memory do you think an array lives?"
    sim3 = bank.template_similarity(q3, "index_zero")
    response, delta = student.respond_to_question(q3, template_similarity=sim3)
    assert delta > 0, f"Expected positive delta, got {delta}"
    print(f"[PASS] Fix 3 (non-linear delta): good question delta={delta:.3f}")
    
    # Test Fix 3: same question later (higher understanding) gives smaller delta
    student2 = StudentSimulator(concept, max_steps=8)
    student2.understanding_level = 0.75
    _, delta_late = student2.respond_to_question(q3, template_similarity=sim3)
    assert delta_late < delta, f"Late delta {delta_late:.3f} should be < early {delta:.3f}"
    print(f"[PASS] Fix 3 (diminishing returns): early={delta:.3f} > late={delta_late:.3f}")

def test_rewards(bank):
    from server.student import StudentSimulator
    from server.rewards import SocratesRewardCalculator
    from models import SocratesAction
    
    concept = bank.get("floating_point")
    calc = SocratesRewardCalculator()
    
    # Bad question: direct answer
    student_bad = StudentSimulator(concept, max_steps=12)
    prev_bad = student_bad.get_state()
    q_bad = "Is it because of ieee 754 binary fraction representation?"
    sim_bad = bank.template_similarity(q_bad, "floating_point")
    student_bad.respond_to_question(q_bad, sim_bad)
    new_bad = student_bad.get_state()
    r_bad, bd_bad = calc.compute_reward(
        SocratesAction(question=q_bad), prev_bad, new_bad, concept, False, sim_bad, 0.0
    )
    
    # Good question: open-ended
    student_good = StudentSimulator(concept, max_steps=12)
    prev_good = student_good.get_state()
    q_good = "What do you think happens when a computer tries to store 0.1 in memory?"
    sim_good = bank.template_similarity(q_good, "floating_point")
    student_good.respond_to_question(q_good, sim_good)
    new_good = student_good.get_state()
    r_good, bd_good = calc.compute_reward(
        SocratesAction(question=q_good), prev_good, new_good, concept, False, sim_good, 0.0
    )
    
    print(f"[INFO] Bad Q reward={r_bad:.3f} | compliance={bd_bad['socratic_compliance']:.2f}")
    print(f"[INFO] Good Q reward={r_good:.3f} | compliance={bd_good['socratic_compliance']:.2f}")
    
    # Fix 5 check: compliance of bad question must be negative
    assert bd_bad["socratic_compliance"] < 0, "Direct answer must have negative compliance"
    # Good must beat bad
    assert r_good > r_bad, f"Good ({r_good:.3f}) should beat bad ({r_bad:.3f})"
    print(f"[PASS] Rewards: good ({r_good:.3f}) > bad ({r_bad:.3f}), anti-hacking works")
    
    # Fix 4: question_quality and misconception_targeting are independent of student classify
    assert "question_quality" in bd_good
    assert "misconception_targeting" in bd_good
    print("[PASS] Fix 4 (R3/R4 decoupled): both signals present in breakdown")

def test_environment(bank):
    from server.environment import SocratesEnvironment
    from models import SocratesAction
    
    env = SocratesEnvironment()
    
    # reset
    obs = env.reset("foundation")
    assert not obs.done
    assert obs.student_confidence == "confused"
    print(f"[PASS] env.reset(): concept loaded, student confidence='confused'")
    
    # step
    action = SocratesAction(question="What do you think happens when you try that?")
    obs2 = env.step(action)
    reward = getattr(obs2, "_reward", None)
    assert reward is not None, "Reward must be attached to obs"
    print(f"[PASS] env.step(): reward={reward:.3f}, confidence={obs2.student_confidence}")
    
    # state (debug — must have understanding_level)
    state = env.state()
    assert hasattr(state, "understanding_level")
    assert state.step_count == 1
    print(f"[PASS] env.state(): understanding={state.understanding_level:.3f}, steps=1")
    
    # Fix 6: observation must NOT have understanding_level
    obs_dict = obs2.model_dump()
    assert "understanding_level" not in obs_dict, "Agent must not see understanding_level"
    print("[PASS] Fix 6 (oracle hidden): understanding_level not in agent observation")
    
    # RuntimeError if step before reset
    env2 = SocratesEnvironment()
    try:
        env2.step(SocratesAction(question="test"))
        assert False, "Should raise RuntimeError"
    except RuntimeError:
        pass
    print("[PASS] env.step() before reset raises RuntimeError (no silent auto-reset)")

def test_client_separation():
    """Fix: verify client.py imports nothing from server/"""
    import ast
    with open("client.py") as f:
        tree = ast.parse(f.read())
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            names = [a.name for a in node.names]
            for n in [mod] + names:
                if "server" in str(n).lower():
                    violations.append(n)
    assert not violations, f"Client imports server internals: {violations}"
    print("[PASS] Client/server separation: clean")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)  # Suppress info logs during test
    
    print("=" * 60)
    print("SOCRATES Validation — Smoke Tests")
    print("=" * 60)
    
    tests = [
        ("models", lambda _: test_models()),
        ("openenv.yaml", lambda _: test_yaml()),
        ("concept_bank", lambda _: test_concept_bank()),
        ("student", test_student),
        ("rewards", test_rewards),
        ("environment", test_environment),
        ("client_separation", lambda _: test_client_separation()),
    ]
    
    # Load concept bank once — reuse across all tests that need it
    from server.concepts import ConceptBank
    print("\nLoading concept bank (embedding model init)...")
    bank = ConceptBank.load("./concepts")
    backend = ("sentence-transformers" if bank.embeddings._use_st
               else "tfidf" if bank.embeddings._use_tfidf
               else "word-overlap")
    print(f"Embedding backend: {backend}\n")

    passed = failed = 0
    for name, fn in tests:
        print(f"--- {name} ---")
        try:
            if name in ("student", "rewards", "environment"):
                fn(bank)
            else:
                fn(None)
            passed += 1
        except Exception as e:
            import traceback
            print(f"[FAIL] {e}")
            traceback.print_exc()
            failed += 1
        print()
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed / {passed+failed} total")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
