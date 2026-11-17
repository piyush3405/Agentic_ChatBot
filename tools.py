import ast
import math
import operator
from dotenv import load_dotenv
from database import save_memory, search_memory
from rag import retrieve_from_rag
load_dotenv()


from langchain_core.tools import tool
from langchain_tavily import TavilySearch

CURRENT_THREAD_ID = "default"


def set_current_thread_id(thread_id: str):
    global CURRENT_THREAD_ID
    CURRENT_THREAD_ID = thread_id


web_search = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced",
)



@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """
    allowed = {
        "math": math,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
    }

    def evaluate(node: ast.AST):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [evaluate(element) for element in node.elts]
            return values if isinstance(node, ast.List) else tuple(values)
        if isinstance(node, ast.BinOp):
            operators = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.FloorDiv: operator.floordiv,
                ast.Mod: operator.mod,
                ast.Pow: operator.pow,
            }
            operation = operators.get(type(node.op))
            if operation is None:
                raise ValueError("Unsupported operator")
            return operation(evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp):
            operators = {ast.UAdd: operator.pos, ast.USub: operator.neg}
            operation = operators.get(type(node.op))
            if operation is None:
                raise ValueError("Unsupported unary operator")
            return operation(evaluate(node.operand))
        if isinstance(node, ast.Call) and not node.keywords:
            if isinstance(node.func, ast.Name):
                function = allowed.get(node.func.id)
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "math"
            ):
                function = getattr(math, node.func.attr, None)
            else:
                function = None
            if not callable(function):
                raise ValueError("Unsupported function")
            return function(*(evaluate(argument) for argument in node.args))
        raise ValueError("Unsupported expression")

    try:
        result = evaluate(ast.parse(expression, mode="eval").body)
        return str(result)
    except Exception as error:
        return f"Calculation error: {str(error)}"

@tool
def remember_this(memory: str) -> str:
    """
    Save an important user preference or fact into long-term memory.
    Use this when the user asks you to remember something.
    """
    return save_memory(
        thread_id=CURRENT_THREAD_ID,
        memory=memory,
    )


@tool
def search_uploaded_documents(query: str) -> str:
    """
    Search uploaded documents for relevant information.
    Use this when the user asks about uploaded PDFs, DOCX, TXT, notes, files, or documents.
    """
    return retrieve_from_rag(
        query=query,
        thread_id=CURRENT_THREAD_ID,
    )


@tool
def recall_memory(query: str) -> str:
    """Recall saved long-term memories about the user or this conversation."""
    return search_memory(
        thread_id=CURRENT_THREAD_ID,
        query=query,
    )


tools = [
    calculator,
    search_uploaded_documents,
    remember_this,
    recall_memory,
    web_search,
]