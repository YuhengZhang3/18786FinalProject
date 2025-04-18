# flight_rating_tool.py
import json, textwrap
from typing import List, Dict

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY


class RatingInput(BaseModel):
    flights: list[dict] = Field(..., description="List of flight dicts")


def _rate_with_llm(flights: List[Dict]) -> str:
    """Call GPT‑4o to sort + rate + add a short comment, return JSON string"""
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.2,
        openai_api_key=OPENAI_API_KEY,
    )

    system = textwrap.dedent("""
        You are an experienced travel agent. 
        Read the list of flights and:
        1. Score each flight on price, duration, and stopovers (0–10, higher is better).
        2. Sort flights from best to worst.
        3. Output ONLY valid JSON with the following structure:
        4. Don't modify the flight information!!!
        {
          "recommended": <best flight dict>,
          "flights": [<sorted flight dicts, each with an added "score">],
          "commentary": "<one‑sentence summary explaining the recommendation>"
        }
    """)

    user = json.dumps({"flights": flights}, ensure_ascii=False)
    response = llm.invoke([("system", system), ("user", user)])
    return response.content   # function‑calling 已保证是 JSON


def rating_tool_bridge(flights: list[dict]) -> str:       # ← 类型对齐
    return _rate_with_llm(flights)


rate_flights_tool = StructuredTool.from_function(
    name="rate_flights",
    description="Given a list of flights, evaluate and rank them.",
    func=rating_tool_bridge,
    args_schema=RatingInput,
    return_direct=True,
)
