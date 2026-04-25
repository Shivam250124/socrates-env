"""Quick audit of all concept JSONs."""
import json, os

for f in sorted(os.listdir("concepts")):
    if not f.endswith(".json"):
        continue
    data = json.load(open(os.path.join("concepts", f)))
    cid = data.get("concept_id", "?")
    ms = data.get("min_steps_to_success", "MISSING")
    rkeys = list(data["responses"].keys())
    templates = len(data.get("good_question_templates", []))
    misconceptions = len(data.get("misconception_phrases", []))
    print(f"{cid:25s} min_steps={str(ms):7s} templates={templates} misconceptions={misconceptions} resp_keys={rkeys}")
