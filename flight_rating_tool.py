# flight_rating_tool.py
import json, textwrap
from typing import List, Dict

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY


class RatingInput(BaseModel):
    flights: list[dict] = Field(..., description="List of flight dicts")

class RatingOutput(BaseModel):
    """What the tool will *return* to the LangGraph runtime."""
    recommended: dict               # ← for the agent
    markdown: str                   # ← for the chat UI

def _rate_with_llm(flights: List[Dict]) -> str:
    """Call GPT‑4o to sort + rate + add a short comment, return JSON string"""
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.2,
        openai_api_key=OPENAI_API_KEY,
    )

    system = """
        You are an experienced travel agent.

        OUTPUT SPEC:
        -------------
        1. First line: a single JSON object with ONLY the recommended flight.
           Example: {"recommended": {...}}   ← must be valid JSON.
        2. Then, a blank line.
        3. Then, a nicely formatted Markdown report (table, bullets, whatever).
        Nothing else.

        Rules:
        1.Keep the book url as it is and show it in both the markdown and JSON.
        2.The nicely formatted Markdown report should first contain the Recommended Flight and the recommend reason, and the rest of the flight info with the booking link (just show 5 of them that you think is also competitive). Tell us the total amount of flights.
        3. Don't make up flights, if there isn't enough flights, just show the flights that are available.
    """

    user = json.dumps({"flights": flights}, ensure_ascii=False)
    response = llm.invoke([("system", system), ("user", user)])
    raw = response.content.strip()

    # split once – JSON part ↑  , Markdown part ↓
    json_line, md = raw.split("\n", 1)
    recommended = json.loads(json_line)["recommended"]

    return RatingOutput(recommended=recommended, markdown=md)


def rating_tool_bridge(flights: list[dict]) -> str:       # ← 类型对齐
    return _rate_with_llm(flights)


rate_flights_tool = StructuredTool.from_function(
    name="rate_flights",
    description="Rank flights and present a Markdown report. "
                "Also returns the best flight in JSON so the agent can use it.",
    func=rating_tool_bridge,
    args_schema=RatingInput,
    return_direct=False,          # ★ let the agent keep thinking after this step
)
