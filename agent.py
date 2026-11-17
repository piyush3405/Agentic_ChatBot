import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
import certifi

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from tools import tools


Path("data").mkdir(exist_ok=True)

ALLOWED_MODELS = {
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
}

_configured_model = os.getenv("GOOGLE_MODEL") or os.getenv("GEMINI_MODEL")
DEFAULT_MODEL = (
    _configured_model
    if _configured_model in ALLOWED_MODELS
    else "gemini-3.5-flash-lite"
)

SYSTEM_PROMPT = """
You are a helpful Agentic AI assistant named Agentic_chatbot similar to ChatGPT.

You can:
1. Answer normal questions.
2. Use tools when needed.
3. Search uploaded documents using the RAG tool.
4. Search the web for latest/current information using Tavily Search.
5. Remember important user information using the memory tool.
6. Recall memory when useful.
7. Use calculator for math.

Rules:
- If the user asks about latest news, current events, recent updates, today's information, or current price information, use web_search.
- If the user asks about an uploaded document, use search_uploaded_documents.
- If the user asks you to remember something, use remember_this.
- If the user asks about previous preferences or saved facts, use recall_memory.
- Use calculator for math questions.
- When using web search, summarize clearly and mention that the answer is based on web search results.
- Be clear, helpful, and concise.
"""


def normalize_model_name(model_name: str | None) -> str:
    """
    Validate selected model from frontend.
    If model is missing or not allowed, fallback to DEFAULT_MODEL.
    """
    if not model_name:
        return DEFAULT_MODEL

    model_name = model_name.strip()

    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL

    return model_name


def build_agent(model_name: str):
    """Build one LangGraph agent with timeouts, retries, and automatic model fallbacks."""
    selected_model = normalize_model_name(model_name)

    # 1. Initialize primary model with a strict timeout and bind tools
    primary_llm = ChatGoogleGenerativeAI(
        model=selected_model,
        temperature=0.3,
        streaming=True,
        timeout=25,  # Prevents infinite hanging if server stalls
    ).bind_tools(tools)

    # 2. Initialize backup free-tier models and bind tools individually
    fallback_models = [m for m in ALLOWED_MODELS if m != selected_model]
    
    fallbacks = [
        ChatGoogleGenerativeAI(
            model=fb_model,
            temperature=0.3,
            streaming=True,
            timeout=25,
        ).bind_tools(tools)
        for fb_model in fallback_models
    ]

    # 3. Combine retries (for brief hiccups) + fallbacks (for 503/429 spikes)
    robust_llm = primary_llm.with_retry(
        stop_after_attempt=3,            # Maximum retry attempts before falling back
        wait_exponential_jitter=True,    # Adds randomized exponential backoff delay
    ).with_fallbacks(fallbacks)

    def chatbot_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

        response = robust_llm.invoke(messages)

        return {
            "messages": [response]
        }

    tool_node = ToolNode(tools)
    workflow = StateGraph(MessagesState)
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", tool_node)
    workflow.add_edge(START, "chatbot")
    workflow.add_conditional_edges("chatbot", tools_condition)
    workflow.add_edge("tools", "chatbot")

    conn = sqlite3.connect(
        "data/langgraph_checkpoints.sqlite",
        check_same_thread=False,
    )
    checkpointer = SqliteSaver(conn)

    return workflow.compile(checkpointer=checkpointer)


_AGENT_CACHE = {}


def get_agent(model_name: str | None = None):
    """
    Return cached LangGraph agent for selected model.
    If not created yet, create it once and reuse it.
    """
    selected_model = normalize_model_name(model_name)

    if selected_model not in _AGENT_CACHE:
        _AGENT_CACHE[selected_model] = build_agent(selected_model)

    return _AGENT_CACHE[selected_model]