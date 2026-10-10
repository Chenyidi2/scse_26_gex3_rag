"""Expose policy lookup as a LangChain tool without loading a model at import."""

from langchain_core.tools import tool

from rag import retrieve


def make_policy_tool(store):
    @tool
    def search_policies(question: str) -> list[dict]:
        """Find three relevant passages in the university IT policy file."""
        return [
            {"id": doc.metadata["policy_id"], "topic": doc.metadata["topic"],
             "source": doc.metadata["source"], "text": doc.page_content}
            for doc in retrieve(question, store)
        ]

    return search_policies
