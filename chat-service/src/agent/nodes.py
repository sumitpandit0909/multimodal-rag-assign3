import logging
from typing import Dict, Any, List
from google import genai
from google.genai import types
from motor.motor_asyncio import AsyncIOMotorDatabase

from agent.state import AgentState
from agent.tools import search_vector_store
from agent.llm import (
    get_openrouter_client,
    generate_with_gemma_openrouter,
    generate_with_gemini_fallback,
    SYSTEM_PROMPT
)

logger = logging.getLogger(__name__)

class RAGNodes:
    """
    Encapsulates all individual graph nodes and transition decisions
    for the LangGraph Agentic RAG workflow.
    """
    def __init__(self, db: AsyncIOMotorDatabase, client: genai.Client):
        self.db = db
        self.client = client

    async def retrieve(self, state: AgentState) -> Dict[str, Any]:
        """Node 1: Retrieve candidate vectors from MongoDB Atlas."""
        query = state.get("query", "")
        logger.info(f"[Node:retrieve] Searching MongoDB Atlas for query: '{query}'")
        sources = await search_vector_store(query, self.db, self.client, top_k=4)
        return {"retrieved_sources": sources}

    async def grade_documents(self, state: AgentState) -> Dict[str, Any]:
        """Node 2: Evaluate semantic relevance of retrieved context against user question."""
        sources = state.get("retrieved_sources", [])
        query = state.get("query", "")

        if not sources:
            logger.info(f"[Node:grade_documents] No sources returned for query '{query}'.")
            return {"documents_relevant": False}

        # Build concise context sample for grading
        snippets = "\n".join([
            f"- {s.get('file_name', 'doc')}: {s.get('text_content', '')[:300]}"
            for s in sources[:3]
        ])

        grader_prompt = (
            f"You are a retrieval evaluator. Does the retrieved context contain information relevant to answering the user question?\n"
            f"User Question: {query}\n\n"
            f"Retrieved Context:\n{snippets}\n\n"
            f"Answer strictly with either 'YES' or 'NO'."
        )

        openrouter_client = get_openrouter_client()
        try:
            if openrouter_client:
                text = generate_with_gemma_openrouter(
                    client=openrouter_client,
                    prompt=grader_prompt,
                    system_prompt="You are a strict evaluator. Answer strictly YES or NO.",
                    temperature=0.0,
                    max_tokens=10
                )
                is_rel = "YES" in text.strip().upper()
            else:
                resp_text = generate_with_gemini_fallback(
                    client=self.client,
                    contents=grader_prompt,
                    config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=10)
                )
                is_rel = "YES" in resp_text.strip().upper()

            logger.info(f"[Node:grade_documents] Evaluated '{query}': {'YES' if is_rel else 'NO'}")
            return {"documents_relevant": is_rel}
        except Exception as e:
            logger.warning(f"[Node:grade_documents] Grading failed ({e}), defaulting to True")
            return {"documents_relevant": True}

    async def rewrite_query(self, state: AgentState) -> Dict[str, Any]:
        """Node 3: Reformulate and expand search query if retrieval was irrelevant."""
        current_query = state.get("query", "")
        orig_query = state.get("original_query", current_query)
        rewrite_count = state.get("rewrite_count", 0) + 1

        prompt = (
            f"The search query '{current_query}' (original intent: '{orig_query}') failed to retrieve relevant documents from the corporate knowledge base. "
            f"Formulate an improved, concise search query that expands technical keywords, removes noise words, and uses synonyms. "
            f"Output ONLY the new query string without quotes or explanation."
        )

        openrouter_client = get_openrouter_client()
        rewritten = current_query
        try:
            if openrouter_client:
                rewritten = generate_with_gemma_openrouter(
                    client=openrouter_client,
                    prompt=prompt,
                    system_prompt="You are a search query optimizer. Output ONLY the query string.",
                    temperature=0.2,
                    max_tokens=50
                ).strip().strip('"').strip("'")
            else:
                rewritten = generate_with_gemini_fallback(
                    client=self.client,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=50)
                ).strip().strip('"').strip("'")
        except Exception as e:
            logger.warning(f"[Node:rewrite_query] Query rewrite failed: {e}")

        logger.info(f"[Node:rewrite_query] (Attempt {rewrite_count}) Rewrote '{current_query}' -> '{rewritten}'")
        return {"query": rewritten, "rewrite_count": rewrite_count}

    async def generate(self, state: AgentState) -> Dict[str, Any]:
        """Node 4: Grounded response synthesis citing sources [1], [2]."""
        query = state.get("original_query") or state.get("query", "")
        sources = state.get("retrieved_sources", [])

        context_str = ""
        for idx, s in enumerate(sources, start=1):
            src_label = f"Page {s['page_number']}" if s.get("page_number") else f"Sheet {s.get('sheet_name')}"
            context_str += f"\n--- Source [{idx}]: {s['file_name']} ({src_label}) ---\n{s['text_content']}\n"

        user_prompt = f"User Question: {query}\n\nContext:\n{context_str}\n\nAnswer with inline citations [1], [2]:"

        openrouter_client = get_openrouter_client()
        answer_text = ""
        if openrouter_client:
            try:
                logger.info("[Node:generate] Generating answer using google/gemma-3-27b-it via OpenRouter...")
                answer_text = generate_with_gemma_openrouter(openrouter_client, user_prompt)
            except Exception as e:
                logger.warning(f"[Node:generate] OpenRouter generation error: {e}")

        if not answer_text:
            answer_text = generate_with_gemini_fallback(
                client=self.client,
                contents=user_prompt
            )

        source_nodes = []
        for idx, s in enumerate(sources, start=1):
            node = {
                "citation_id": idx,
                "file_name": s["file_name"],
                "source_type": s["source_type"]
            }
            if s["source_type"] == "visual":
                node["screenshot_url"] = s.get("screenshot_url")
                node["page_number"] = s.get("page_number")
            elif s["source_type"] == "tabular":
                node["sheet_name"] = s.get("sheet_name")
                node["raw_data"] = s.get("raw_data")
            source_nodes.append(node)

        return {"answer": answer_text, "source_nodes": source_nodes}

    async def no_sources(self, state: AgentState) -> Dict[str, Any]:
        """Node 5: Polite notification when no relevant records exist."""
        query = state.get("original_query") or state.get("query", "")
        no_sources_prompt = (
            f"User asked: '{query}'\n"
            "Inform the user politely that no matching documents or vector records were found in the enterprise knowledge base. "
            "If documents have not been ingested yet, invite them to drop files into the ingestion service."
        )
        openrouter_client = get_openrouter_client()
        if openrouter_client:
            try:
                ans = generate_with_gemma_openrouter(openrouter_client, no_sources_prompt)
                return {"answer": ans, "source_nodes": []}
            except Exception as e:
                logger.warning(f"[Node:no_sources] OpenRouter error: {e}")

        resp_text = generate_with_gemini_fallback(
            client=self.client,
            contents=no_sources_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3
            )
        )
        return {"answer": resp_text, "source_nodes": []}

    @staticmethod
    def decide_after_grading(state: AgentState) -> str:
        """Conditional routing evaluator after grade_documents."""
        is_relevant = state.get("documents_relevant", False)
        rewrite_count = state.get("rewrite_count", 0)
        sources = state.get("retrieved_sources", [])

        if is_relevant:
            return "generate"

        # If not relevant and haven't rewritten yet, trigger rewrite loop
        if rewrite_count < 1:
            return "rewrite"

        # If already rewritten and still not relevant, use fallback logic
        if sources:
            return "generate"
        else:
            return "no_sources"
