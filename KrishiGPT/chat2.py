"""
chat2.py
--------
Retrieval-Augmented Generation (RAG) chain for KrishiGPT.

Fully open-source / free stack:
  * Generator LLM : an open GPT-architecture model (Llama family) served
                    locally through Ollama -- no paid API, no keys.
  * Retriever     : a FAISS vector store (built in chat1.py).

Prompts are written so the model answers in the SAME language the farmer
asked in (Marathi or English) -- see rag/multilingual.py for the
detection/translation helpers.
"""

import os

from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# Language model (open-source, GPT-architecture, served by Ollama)
# ---------------------------------------------------------------------------
# Install Ollama (https://ollama.com) then:  ollama pull llama3.2
# Everything runs locally and free of cost. Override with KRISHIGPT_MODEL.
LLM_MODEL = os.getenv("KRISHIGPT_MODEL", "llama3.2")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm = Ollama(
    model=LLM_MODEL,
    base_url=OLLAMA_URL,
    temperature=0.1,
    num_predict=512,
)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
# {answer_language} is filled in per-request so the model replies in Marathi
# or English to match the farmer's question.
PROMPT_TEMPLATE = """You are KrishiGPT, a crop-advisory assistant for Indian farmers.
Answer questions about agriculture -- crops, soil, irrigation, pests, diseases,
weather and government schemes. Explain in simple words, in under 120 words.
Reply ENTIRELY in {answer_language}. If the context does not contain the
answer, say you do not know (in {answer_language}).

CONTEXT:
{context}

QUESTION: {question}

ANSWER (in {answer_language}):"""


def setup_retrieval_qa(db, answer_language: str = "English"):
    """Build a RetrievalQA chain over ``db`` that answers in ``answer_language``.

    Args:
        db: a FAISS vector store returned by ``initialize_vector_store``.
        answer_language: human-readable language name ("Marathi" / "English").
    """
    retriever = db.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
        partial_variables={"answer_language": answer_language},
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=False,
    )
    return chain
