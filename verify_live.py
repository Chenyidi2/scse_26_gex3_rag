"""Record real local-model responses separately from offline unit tests."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import urllib.request

from agent import Answer, MODEL, Review, UNKNOWN, make_model, review_answer, run_question
from rag import EMBEDDING_MODEL, POLICY_FILE, base_url, open_store

CASES = [
    ("missing_phone", "What exact phone number should I call for the University IT Service Desk?", None),
    ("print_retention", "After how many hours are unreleased print jobs automatically deleted?", r"\b24\s+hours?\b"),
    ("visitor_wifi", "Which wireless network should guests use?", r"\bCampus-Guest\b"),
    ("account_lock", "How long should I wait before trying a temporarily locked account again?", r"\b15\s+minutes?\b"),
]


def loaded_models():
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with client.open(base_url() + "/api/ps", timeout=10) as response:
        return json.load(response)["models"]


def gpu_entry(model):
    state = next((entry for entry in loaded_models() if entry["name"] == model), None)
    if not state or state.get("size_vram", 0) <= 0 or state["size_vram"] != state.get("size"):
        raise RuntimeError(f"{model} did not report full GPU offloading")
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write("{}")
    result = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "policy_sha256": hashlib.sha256(POLICY_FILE.read_bytes()).hexdigest(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("langchain-core", "langchain-ollama", "langchain-chroma", "pydantic")},
        "cases": [], "review_controls": [],
    }

    def save():
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        store = open_store()
        writer, reviewer = make_model(Answer), make_model(Review)
        for name, question, expected in CASES:
            # The expected value is used only after generation, never in its prompt.
            vector = store.embeddings.embed_query(question)
            if len(vector) != 768 or not all(math.isfinite(value) for value in vector):
                raise RuntimeError("Unexpected embedding dimensions")
            embedding_state = gpu_entry(EMBEDDING_MODEL)
            report = run_question(question, store, writer, reviewer)
            report["case"] = name
            result["cases"].append(report)
            save()
            report["embedding_gpu"] = embedding_state
            report["chat_gpu"] = gpu_entry(MODEL)
            response = report.get("answer", "")
            report["check_passed"] = report["status"] == "accepted" and (
                (response == UNKNOWN and report["sources"] == []) if expected is None
                else (bool(re.search(expected, response, re.IGNORECASE)) and bool(report.get("sources")))
            )
            save()
            print(f"{name}: {'PASS' if report['check_passed'] else 'FAIL'}: {response or report.get('error')}", flush=True)
        for source_case, wrong_answer, name in (
            ("visitor_wifi", "Guests should use Campus-Secure.", "incorrect_network"),
            ("print_retention", UNKNOWN, "unnecessary_unknown"),
        ):
            source = next(case for case in result["cases"] if case["case"] == source_case)
            passages = source["trace"][0]["passages"]
            candidate = Answer(answer=wrong_answer, sources=[] if wrong_answer == UNKNOWN else [passages[0]["id"]])
            trace = []
            control = {"name": name, "question": source["question"], "candidate": candidate.model_dump(),
                       "trace": trace, "check_passed": False}
            result["review_controls"].append(control)
            verdict = review_answer(source["question"], passages, candidate, reviewer, trace, "negative_control")
            control.update(verdict=verdict.model_dump(), check_passed=verdict.verdict == "REVISE")
            save()
            print(f"review {name}: {'PASS' if control['check_passed'] else 'FAIL'}", flush=True)
        result["status"] = "passed" if all(case["check_passed"] for case in result["cases"] + result["review_controls"]) else "failed"
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
