import datetime, json, os
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver          # ### NEW

from config import OPENAI_API_KEY
from flight_search import SkyscannerFlightSearchTool


# ----------------------------------------------------------------------
# 1.  Keep your original flight‑search tool
# ----------------------------------------------------------------------
flight_search = SkyscannerFlightSearchTool()


# ----------------------------------------------------------------------
# 2.  Strict input schema (unchanged)
# ----------------------------------------------------------------------
class FlightSearchInput(BaseModel):
    origin: str = Field(..., description="IATA code or city name of the departure location")
    destination: str = Field(..., description="IATA code or city name of the arrival location")
    departure_date: str = Field(..., description="YYYY‑MM‑DD")
    return_date: Optional[str] = Field(None, description="YYYY‑MM‑DD if round‑trip")
    adults: int = Field(1, ge=1, le=9)
    cabin_class: str = Field(
        "Economy",
        description="Economy | Premium Economy | Business | First"
    )
    children: int = 0
    infants: int = 0

    @field_validator("departure_date", "return_date")
    @classmethod
    def validate_dates(cls, v):
        if v:
            datetime.datetime.strptime(v, "%Y-%m-%d")
        return v


# ----------------------------------------------------------------------
# 3.  Wrap the tool
# ----------------------------------------------------------------------
def flight_search_bridge(**kwargs):
    return flight_search.invoke(json.dumps(kwargs))

flight_tool = StructuredTool.from_function(
    name="flight_search",
    description="Search flights via Skyscanner",
    func=flight_search_bridge,
    args_schema=FlightSearchInput,
    return_direct=True,
)

# ----------------------------------------------------------------------
# 4.  LLM
# ----------------------------------------------------------------------
llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.5,
    openai_api_key=OPENAI_API_KEY
)

# ----------------------------------------------------------------------
# 5.  System prompt
# ----------------------------------------------------------------------
today = datetime.datetime.now().strftime("%Y-%m-%d")
SYSTEM_PROMPT = f"""
Today is {today}. You are a flight‑booking assistant.

RULES:
1. Use **exactly** the origin and destination strings the user provides.
2. If a city is ambiguous or unrecognized, ASK the user for clarification instead of guessing.
3. Convert relative dates like "tomorrow" to YYYY‑MM‑DD.
4. After you have all REQUIRED fields (origin, destination, departure_date), call the flight_search function.
5. Tell the user your Flight Search Input and ask them to recheck it when you are unsure about it.

Memory rules:
• Conversation history is reliable. If the user has already given a value for
  origin or destination, you may reuse it without asking again—unless the user
  explicitly changes it.
• Only ask follow‑up questions for the fields that are still UNKNOWN after
  checking the history.
"""

# ----------------------------------------------------------------------
# 6.  Build graph WITH memory
# ----------------------------------------------------------------------
memory_store = MemorySaver()                                  # ### NEW
TOOLS = [flight_tool]

graph = create_react_agent(
    model=llm,
    tools=TOOLS,
    prompt=SYSTEM_PROMPT,
    checkpointer=memory_store,                                 # ### NEW
)

# ----------------------------------------------------------------------
# 7.  CLI loop – keep a constant thread_id so state persists
# ----------------------------------------------------------------------
def main() -> None:
    print("🛫  Flight Search AI Agent  |  type 'exit' to quit")
    thread_id = "local-session"                               # ### NEW

    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        inputs = {"messages": [("user", user)]}
        config = {"configurable": {"thread_id": thread_id}}   # ### NEW

        result = graph.invoke(inputs, config=config)          # ### NEW
        assistant_reply = result["messages"][-1].content
        print("Bot:", assistant_reply, "\n")


if __name__ == "__main__":
    main()
