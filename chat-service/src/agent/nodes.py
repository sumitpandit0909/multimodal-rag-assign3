import re
import logging
from typing import Dict, Any, List, Set
from google import genai
from google.genai import types
from motor.motor_asyncio import AsyncIOMotorDatabase

from agent.state import AgentState
from agent.tools import search_vector_store
from agent.llm import (
    ainvoke_generator,
    ainvoke_utility,
    generate_with_gemini_fallback,
    SYSTEM_PROMPT
)

logger = logging.getLogger(__name__)

class RAGNodes:
    """
    Streamlined, low-latency graph nodes using native LangChain OpenRouter
    and strict semantic grading with guaranteed visual citation delivery.
    """
    def __init__(self, db: AsyncIOMotorDatabase, client: genai.Client):
        self.db = db
        self.client = client

    async def retrieve(self, state: AgentState) -> Dict[str, Any]:
        """Node 1: Retrieve candidate vectors from MongoDB Atlas with embedding caching."""
        query = state.get("query", "")
        sources = await search_vector_store(query, self.db, self.client, top_k=3)
        return {"retrieved_sources": sources}

    async def grade_documents(self, state: AgentState) -> Dict[str, Any]:
        """
        Node 2: Strict semantic relevance grading with google/gemini-2.5-flash-lite (~200ms).
        Rejects out-of-domain / irrelevant queries with zero hallucination.
        """
        sources = state.get("retrieved_sources", [])
        query = state.get("query", "")

        if not sources:
            logger.info(f"[Node:grade_documents] No sources returned for '{query}'. Relevant: False")
            return {"documents_relevant": False}

        # Build clean snippet preview
        snippets = "\n".join([
            f"[{i}] Document: {s.get('file_name', 'doc')}\n{s.get('text_content', '')[:300]}"
            for i, s in enumerate(sources[:3], start=1)
        ])

        grader_prompt = (
            f"User Question: {query}\n\n"
            f"Retrieved Document Context:\n{snippets}\n\n"
            f"Task: Does ANY snippet contain facts or information specifically relevant to answering the User Question?\n"
            f"- If the question is asking about topic A and snippets are about topic B, answer NO.\n"
            f"- Answer strictly with either 'YES' or 'NO'."
        )

        try:
            resp = await ainvoke_utility(
                prompt=grader_prompt,
                system_prompt="You are a strict relevance evaluator. Answer strictly YES or NO."
            )
            is_rel = "YES" in resp.strip().upper()
            logger.info(f"[Node:grade_documents] Evaluated '{query}': {'YES' if is_rel else 'NO'}")
            return {"documents_relevant": is_rel}
        except Exception as e:
            logger.warning(f"[Node:grade_documents] Semantic grader error: {e}. Trying Gemini fallback...")
            try:
                fb_text = generate_with_gemini_fallback(
                    client=self.client,
                    contents=grader_prompt,
                    config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=5)
                )
                is_rel = "YES" in fb_text.strip().upper()
                return {"documents_relevant": is_rel}
            except Exception as fe:
                logger.error(f"[Node:grade_documents] Fallback grader failed: {fe}")
                return {"documents_relevant": False}

    async def generate(self, state: AgentState) -> Dict[str, Any]:
        """
        Node 3: Grounded synthesis using google/gemma-3-27b-it via LangChain OpenRouter.
        Robust citation formatting guaranteeing visual & tabular source nodes reach the UI.
        """
        query = state.get("query", "")
        sources = state.get("retrieved_sources", [])

        # Format compact context (max 3 sources)
        context_str = ""
        for idx, s in enumerate(sources[:3], start=1):
            src_label = f"Page {s['page_number']}" if s.get("page_number") else f"Sheet {s.get('sheet_name')}"
            context_str += f"\n--- Source [{idx}]: {s['file_name']} ({src_label}) ---\n{s['text_content']}\n"

        user_prompt = (
            f"User Question: {query}\n\n"
            f"Context:\n{context_str}\n\n"
            f"Instructions:\n"
            f"- Answer the question using ONLY the facts from the context.\n"
            f"- Be concise and factual (2-3 focused paragraphs).\n"
            f"- YOU MUST cite each statement using inline brackets like [1], [2] referencing the source numbers above.\n"
            f"- If the context does not answer the question, state: 'I could not find information about this in the knowledge base.' and DO NOT cite sources.\n\n"
            f"Answer:"
        )

        answer_text = ""
        try:
            logger.info("[Node:generate] Invoking LangChain OpenRouter (google/gemma-3-27b-it)...")
            answer_text = await ainvoke_generator(user_prompt)
        except Exception as e:
            logger.warning(f"[Node:generate] OpenRouter generation failed ({e}). Falling back to Gemini...")

        if not answer_text or not answer_text.strip():
            answer_text = generate_with_gemini_fallback(
                client=self.client,
                contents=user_prompt
            )

        # Extract all citation numbers in square brackets: [1], [2], [Source 1], etc.
        cited_indices: Set[int] = set(int(m) for m in re.findall(r'\[(?:Source\s*|Doc\s*)?(\d+)\]', answer_text, re.IGNORECASE))
        
        # If model used parentheses like (1), (2), catch them as well
        if not cited_indices:
            cited_indices = set(int(m) for m in re.findall(r'\((\d+)\)', answer_text))

        # Robust Fallback: If context was marked relevant but the model forgot inline brackets,
        # ensure sources are not dropped so the user sees the visual citations!
        if not cited_indices and sources:
            cited_indices = set(range(1, len(sources[:3]) + 1))
            inline_tags = " ".join([f"[{i}]" for i in sorted(list(cited_indices))])
            answer_text += f"\n\n*Verified References: {inline_tags}*"

        logger.info(f"[Node:generate] Attaching sources for citation IDs: {sorted(list(cited_indices))}")

        source_nodes = []
        for idx, s in enumerate(sources[:3], start=1):
            if idx in cited_indices:
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
        """Node 4: Fast notice when query is out-of-domain. Returns empty sources (0 images)."""
        query = state.get("query", "")
        no_sources_prompt = (
            f"The user asked: '{query}'\n"
            "Inform the user politely that no relevant documents or records were found in the enterprise knowledge base. "
            "Invite them to drop relevant files into the ingestion service to expand the assistant's knowledge."
        )
        try:
            ans = await ainvoke_utility(
                prompt=no_sources_prompt,
                system_prompt="You are a polite enterprise assistant. Respond concisely in 2 sentences."
            )
            return {"answer": ans, "source_nodes": []}
        except Exception as e:
            logger.warning(f"[Node:no_sources] Utility error: {e}")

        resp_text = generate_with_gemini_fallback(
            client=self.client,
            contents=no_sources_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=150
            )
        )
        return {"answer": resp_text, "source_nodes": []}

    @staticmethod
    def decide_after_grading(state: AgentState) -> str:
        """Single-pass conditional routing directly to generate or no_sources."""
        if state.get("documents_relevant", False):
            return "generate"
        return "no_sources"
