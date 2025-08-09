import json
from backend.semantics.retriever import search
from backend.services.supabase_service import supabase
import cohere
import os

COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

def reason_with_memory(agent_name: str, query: str, namespace: str, k: int = 5) -> str:
    """
    Use embeddings from Supabase + Cohere for reasoning.
    This gives the agent context and lets it decide how to answer.
    """
    # 1. Search semantic memory for relevant context
    hits = search(namespace, query, k=k)

    # 2. Turn context into a readable text blob
    context_text = "\n".join(
        f"[{h.get('created_at','')}]: {h.get('content','')}" for h in hits
    )

    # 3. Ask Cohere to reason over the context
    prompt = f"""
You are the {agent_name} agent. 
Your job is to answer the user's question based on both your reasoning ability and the memory context below.

Memory context:
{context_text}

User's question: {query}

If the answer is in the memory, cite it directly.
If it is not, explain what you know or suggest a next step.
Be concise but complete.
    """
    resp = co.generate(
        model="command-r-plus",  # cheaper than OpenAI
        prompt=prompt,
        max_tokens=300,
        temperature=0.3
    )
    return resp.generations[0].text.strip()
