import hashlib
import json
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import agent
import rag
from rag_tool import make_policy_tool


PASSAGE = {"id": "doc-0010", "topic": "printing", "source": "university_it_policies.txt",
           "text": "Unreleased print jobs are automatically deleted after 24 hours."}


class Search:
    def invoke(self, value):
        return [PASSAGE]


class Replies:
    def __init__(self, *values):
        self.values = iter(values)

    def invoke(self, messages):
        self.messages = messages
        value = next(self.values)
        return AIMessage(content=json.dumps(value), response_metadata={"done_reason": "stop"})

    def bind(self, **kwargs):
        self.bound = kwargs
        return self


def answer(text="After 24 hours.", sources=None):
    return {"answer": text, "sources": ["doc-0010"] if sources is None else sources}


def review(verdict="PASS", reason="The cited passage supports the answer."):
    return {"verdict": verdict, "reason": reason}


class FakeStore:
    def __init__(self):
        self.docs = {}
        self.search_calls = []

    def get(self, include):
        return {"ids": list(self.docs), "documents": [d.page_content for d in self.docs.values()],
                "metadatas": [d.metadata for d in self.docs.values()]}

    def add_documents(self, docs, ids):
        self.docs.update(zip(ids, docs))

    def similarity_search(self, question, k):
        self.search_calls.append((question, k))
        return list(self.docs.values())[:k]


class PipelineTests(unittest.TestCase):
    def test_embedding_constructor_matches_installed_api(self):
        with patch("rag.Chroma") as chroma:
            rag.open_store("unused-index")
        embedding = chroma.call_args.kwargs["embedding_function"]
        self.assertEqual(embedding.keep_alive, 120)
        self.assertEqual(embedding.num_gpu, 99)
        self.assertEqual(embedding.model, rag.EMBEDDING_MODEL)

    def test_chat_constructor_matches_installed_api(self):
        model = agent.make_model(agent.Answer)
        self.assertEqual(model.keep_alive, 120)
        self.assertEqual(model.model, "qwen3:0.6b")
        self.assertTrue(model.reasoning)

    def test_supplied_files_unchanged(self):
        manifest = json.loads((rag.ROOT / "source_manifest.json").read_text())
        for name, expected in manifest["sha256"].items():
            actual = hashlib.sha256((rag.ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            self.assertEqual(expected, actual, name)

    def test_exact_source_conversion(self):
        rows = [json.loads(line) for line in (rag.ROOT / "original_domain_corpus_clean.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), len(rag.documents))
        for row, doc in zip(rows, rag.documents):
            self.assertEqual((row["id"], row["topic"], row["text"].strip()),
                             (doc.metadata["policy_id"], doc.metadata["topic"], doc.page_content))

    def test_index_build_resumes_and_is_idempotent(self):
        store = FakeStore()
        doc = rag.documents[0]
        store.docs[doc.metadata["policy_id"]] = doc
        with patch("builtins.print"):
            rag.build_index(store)
            rag.build_index(store)
        self.assertEqual(len(store.docs), len(rag.documents))

    def test_reject_foreign_index_content(self):
        store = FakeStore()
        store.docs["foreign"] = Document(page_content="unrelated", metadata={})
        with self.assertRaises(ValueError):
            rag.missing_documents(store)

    def test_reject_modified_policy_in_index(self):
        store = FakeStore()
        doc = rag.documents[0]
        store.docs[doc.metadata["policy_id"]] = Document(page_content="changed", metadata=doc.metadata)
        with self.assertRaises(ValueError):
            rag.missing_documents(store)

    def test_reject_partial_index_search(self):
        with self.assertRaises(RuntimeError):
            rag.retrieve("Printing?", FakeStore())

    def test_blank_question(self):
        with self.assertRaises(ValueError):
            rag.retrieve(" ", FakeStore())

    def test_tool_searches_three_and_keeps_sources(self):
        store = FakeStore()
        store.docs = {d.metadata["policy_id"]: d for d in rag.documents}
        result = make_policy_tool(store).invoke({"question": "Printing?"})
        self.assertEqual(store.search_calls, [("Printing?", 3)])
        self.assertEqual(result[0]["text"], rag.documents[0].page_content)
        self.assertEqual(result[0]["source"], "university_it_policies.txt")

    def test_supported_answer(self):
        trace = []
        result = agent.answer_question("When?", Search(), Replies(answer()), Replies(review()), trace)
        self.assertEqual(result.answer, "After 24 hours.")
        self.assertEqual(trace[-1]["stage"], "accepted")

    def test_generation_schema_only_allows_retrieved_ids(self):
        writer = Replies(answer())
        agent.answer_question("When?", Search(), writer, Replies(review()), [])
        sources = writer.bound["format"]["properties"]["sources"]
        self.assertEqual(sources["maxItems"], 1)
        self.assertEqual(sources["items"]["enum"], ["doc-0010"])

    def test_model_unknown_answer(self):
        result = agent.answer_question("Phone?", Search(),
                                       Replies(answer(agent.UNKNOWN, [])),
                                       Replies({"available": False, "quote": ""}), [])
        self.assertEqual(result.answer, agent.UNKNOWN)

    def test_unknown_revised_when_requested_fact_exists(self):
        result = agent.answer_question("When?", Search(),
            Replies(answer(agent.UNKNOWN, []), answer()),
            Replies({"available": True, "quote": PASSAGE["text"]}, review()), [])
        self.assertEqual(result.answer, "After 24 hours.")

    def test_availability_quote_must_be_in_retrieved_text(self):
        with self.assertRaises(ValueError):
            agent.review_answer("When?", [PASSAGE], agent.Answer(**answer(agent.UNKNOWN, [])),
                                Replies({"available": True, "quote": "A made up policy."}), [], "review")

    def test_availability_schema_only_quotes_retrieved_text(self):
        reviewer = Replies({"available": False, "quote": ""})
        agent.review_answer("Phone?", [PASSAGE], agent.Answer(**answer(agent.UNKNOWN, [])),
                            reviewer, [], "review")
        self.assertEqual(reviewer.bound["format"]["properties"]["quote"]["enum"], ["", PASSAGE["text"]])

    def test_factual_review_only_receives_cited_passages(self):
        reviewer = Replies(review())
        extra = {**PASSAGE, "id": "doc-9999", "text": "Uncited unrelated material."}
        agent.review_answer("When?", [PASSAGE, extra], agent.Answer(**answer()), reviewer, [], "review")
        self.assertIn(PASSAGE["text"], reviewer.messages[1].content)
        self.assertNotIn(extra["text"], reviewer.messages[1].content)

    def test_unavailable_fact_cannot_have_a_quote(self):
        with self.assertRaises(ValueError):
            agent.review_answer("When?", [PASSAGE], agent.Answer(**answer(agent.UNKNOWN, [])),
                                Replies({"available": False, "quote": PASSAGE["text"]}), [], "review")

    def test_revision_and_second_review(self):
        trace = []
        result = agent.answer_question("When?", Search(),
                Replies(answer("After 48 hours."), answer()),
                Replies(review("REVISE", "48 is unsupported."), review()), trace)
        self.assertEqual(result.answer, "After 24 hours.")
        self.assertEqual([r["stage"] for r in trace].count("revision"), 1)
        self.assertTrue(any(r["stage"] == "review_2" for r in trace))

    def test_local_check_can_override_model_pass(self):
        with self.assertRaises(ValueError):
            agent.answer_question("When?", Search(),
                Replies(answer("After 48 hours."), answer("After 48 hours.")),
                Replies(review(), review()), [])

    def test_reviewer_failure_never_becomes_unknown(self):
        with self.assertRaises(ValueError):
            agent.answer_question("When?", Search(), Replies(answer(), answer()),
                                 Replies(review("REVISE", "Wrong."), review("REVISE", "Wrong.")), [])

    def test_invented_reference(self):
        self.assertTrue(agent.check_answer(agent.Answer(**answer(sources=["doc-9999"])), [PASSAGE]))

    def test_empty_citations(self):
        self.assertTrue(agent.check_answer(agent.Answer(**answer(sources=[])), [PASSAGE]))

    def test_unknown_with_citation(self):
        self.assertTrue(agent.check_answer(agent.Answer(**answer(agent.UNKNOWN)), [PASSAGE]))

    def test_duplicate_citation(self):
        self.assertTrue(agent.check_answer(agent.Answer(**answer(sources=["doc-0010"] * 2)), [PASSAGE]))

    def test_number_substring_is_not_evidence(self):
        self.assertTrue(agent.check_answer(agent.Answer(**answer("After 4 hours.")), [PASSAGE]))

    def test_email_is_not_in_evidence(self):
        self.assertTrue(agent.check_answer(agent.Answer(**answer("Contact help@example.edu.")), [PASSAGE]))

    def test_invalid_schema_is_rejected_and_raw_is_kept(self):
        trace = []
        with self.assertRaises(ValueError):
            agent.answer_question("When?", Search(), Replies({"answer": "Hello"}), Replies(), trace)
        self.assertEqual(json.loads(trace[-1]["raw"]), {"answer": "Hello"})

    def test_truncated_response_rejected(self):
        model = unittest.mock.Mock()
        model.invoke.return_value = AIMessage(content=json.dumps(answer()),
                                             response_metadata={"done_reason": "length"})
        trace = []
        with self.assertRaises(ValueError):
            agent.invoke_recorded(model, agent.Answer, "system", "prompt", "draft", trace)
        self.assertEqual(trace[-1]["metadata"]["done_reason"], "length")

    def test_service_failure_is_recorded_not_answered(self):
        model = unittest.mock.Mock()
        model.bind.return_value = model
        model.invoke.side_effect = ConnectionError("Service unavailable")
        with patch("agent.make_policy_tool", return_value=Search()):
            report = agent.run_question("When?", None, model, model)
        self.assertEqual(report["status"], "failed")
        self.assertNotIn("answer", report)
        self.assertIn("ConnectionError", report["error"])


if __name__ == "__main__":
    unittest.main()
