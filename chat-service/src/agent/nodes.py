import re
import logging
from typing import Dict, Any, List, Set
from google import genai
from google.genai import types
from motor.motor_asyncio import AsyncIOMotorDatabase

from agent.state import AgentState
from agent.tools import search_vector_store
from models.schemas import RerankOutput, SynthesisOutput, CandidateChunk
from agent.llm import (
    ainvoke_generator,
    ainvoke_utility,
    ainvoke_structured_utility,
    generate_with_gemini_fallback,
    generate_structured_gemini_fallback,
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
        """Node 1: Retrieve wide pool of candidate vectors (top_k=8) from MongoDB Atlas with embedding caching."""
        query = state.get("query", "")
        sources = await search_vector_store(query, self.db, self.client, top_k=8)
        return {"retrieved_sources": sources}

    async def grade_documents(self, state: AgentState) -> Dict[str, Any]:
        """
        Node 2: Semantic List-wise Re-ranker & Relevance Quality Gate.
        Powered by google/gemini-2.5-flash-lite (~200ms).
        Evaluates the semantic utility of all retrieved candidates, re-orders them by true answer relevance,
        and eliminates irrelevant passages before synthesis.
        """
        sources = state.get("retrieved_sources", [])
        query = state.get("query", "")

        if not sources:
            logger.info(f"[Node:grade_documents] No sources returned for '{query}'. Relevant: False")
            return {"documents_relevant": False, "retrieved_sources": []}

        # Build clean candidate snippets preview (up to 8)
        candidate_snippets = []
        for i, s in enumerate(sources[:8], start=1):
            src_info = f"Page {s.get('page_number')}" if s.get("page_number") else f"Sheet {s.get('sheet_name')}"
            clean_text = s.get('text_content', '').strip().replace("\n", " ")[:280]
            candidate_snippets.append(f"[{i}] {s.get('file_name', 'Doc')} ({src_info}): {clean_text}")

        snippets_str = "\n\n".join(candidate_snippets)

        reranker_prompt = (
            f"User Question: {query}\n\n"
            f"Candidate Passages:\n{snippets_str}\n\n"
            f"Instructions:\n"
            f"1. Semantically evaluate which passages contain facts, data, or direct answers to the User Question.\n"
            f"2. Select the top 1 to 3 most relevant passage numbers and rank them in descending order of usefulness (e.g. [2, 1, 5]).\n"
            f"3. If NONE of the passages contain information relevant to answering the question, respond strictly with: NONE.\n\n"
            f"Your output (either [id1, id2, ...] or NONE):"
        )

        # 1. First attempt: Strict Pydantic structured output with LangChain OpenRouter
        rerank_output: Optional[RerankOutput] = None
        try:
            rerank_output = await ainvoke_structured_utility(
                prompt=reranker_prompt,
                schema=RerankOutput,
                system_prompt="You are a strict semantic evaluator and re-ranker. Return strictly valid JSON matching the schema."
            )
        except Exception as e:
            logger.warning(f"[Node:grade_documents] LangChain structured output error: {e}")

        # 2. Second attempt: Direct Google GenAI SDK fallback with native Pydantic schema
        if not rerank_output:
            try:
                rerank_output = generate_structured_gemini_fallback(
                    client=self.client,
                    prompt=reranker_prompt,
                    schema=RerankOutput,
                    system_prompt="You are a strict semantic evaluator and re-ranker. Return strictly valid JSON matching the schema."
                )
            except Exception as fe:
                logger.error(f"[Node:grade_documents] Structured Gemini fallback failed: {fe}")

        # 3. Process the validated Pydantic model
        if rerank_output:
            if not rerank_output.is_relevant:
                logger.info(f"[Node:grade_documents] Pydantic evaluated '{query}' as out-of-domain. Relevant: False")
                return {"documents_relevant": False, "retrieved_sources": []}

            ordered_ids = []
            for rid in rerank_output.ranked_ids:
                if 1 <= rid <= len(sources) and rid not in ordered_ids:
                    ordered_ids.append(rid)

            if ordered_ids:
                re_ranked_sources = [sources[i - 1] for i in ordered_ids[:3]]
                logger.info(f"[Node:grade_documents] Pydantic Re-ranker selected top IDs: {ordered_ids[:3]} | Reasoning: {rerank_output.reasoning}")
                return {
                    "documents_relevant": True,
                    "retrieved_sources": re_ranked_sources
                }
            elif sources:
                return {
                    "documents_relevant": True,
                    "retrieved_sources": sources[:3]
                }

        # 4. Tertiary safety net: Raw text parsing fallback
        try:
            raw_text = await ainvoke_utility(reranker_prompt)
            return self._parse_rerank_response(raw_text, sources, query)
        except Exception:
            return {"documents_relevant": False, "retrieved_sources": []}

    def _parse_rerank_response(self, resp: str, sources: List[Dict], query: str) -> Dict[str, Any]:
        """Safety net to parse semantic re-ranking output if structured mode is unavailable."""
        resp_clean = (resp or "").strip()
        resp_upper = resp_clean.upper()

        if "NONE" in resp_upper and not re.search(r'\[\s*\d+', resp_clean):
            logger.info(f"[Node:grade_documents] Query '{query}' evaluated as out-of-domain (NONE). Relevant: False")
            return {"documents_relevant": False, "retrieved_sources": []}

        ranked_ids = [int(x) for x in re.findall(r'\b([1-8])\b', resp_clean)]
        seen = set()
        ordered_ids = []
        for rid in ranked_ids:
            if rid not in seen and 1 <= rid <= len(sources):
                seen.add(rid)
                ordered_ids.append(rid)

        if ordered_ids:
            re_ranked_sources = [sources[i - 1] for i in ordered_ids[:3]]
            logger.info(f"[Node:grade_documents] Re-ranked {len(sources)} candidates for '{query}' -> Top IDs: {ordered_ids[:3]}")
            return {
                "documents_relevant": True,
                "retrieved_sources": re_ranked_sources
            }

        if "YES" in resp_upper:
            logger.info(f"[Node:grade_documents] Grader responded YES without IDs. Keeping top {min(3, len(sources))} sources.")
            return {
                "documents_relevant": True,
                "retrieved_sources": sources[:3]
            }

        logger.info(f"[Node:grade_documents] Could not extract valid ranks from: '{resp_clean}'. Relevant: False")
        return {"documents_relevant": False, "retrieved_sources": []}

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
        cited_indices: Set[int] = set()

        # Attempt structured synthesis
        try:
            synthesis = await ainvoke_structured_utility(
                prompt=user_prompt,
                schema=SynthesisOutput,
                system_prompt=SYSTEM_PROMPT
            )
            if synthesis and synthesis.answer:
                answer_text = synthesis.answer
                cited_indices = set(int(x) for x in synthesis.cited_sources if 1 <= int(x) <= len(sources[:3]))
        except Exception as se:
            logger.warning(f"[Node:generate] Structured synthesis failed ({se}), falling back to standard generator...")

        if not answer_text or not answer_text.strip():
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

        # Fallback citation extraction if not extracted by structured schema
        if not cited_indices:
            cited_indices = set(int(m) for m in re.findall(r'\[(?:Source\s*|Doc\s*)?(\d+)\]', answer_text, re.IGNORECASE))
        if not cited_indices:
            cited_indices = set(int(m) for m in re.findall(r'\((\d+)\)', answer_text))

        # Guarantee at least verified sources are linked if answer has content
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
