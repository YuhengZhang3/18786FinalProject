import datetime, json, os
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver          # ### NEW

from config import OPENAI_API_KEY
from flight_search import SkyscannerFlightSearchTool
from flight_rating_tool import rate_flights_tool
from langchain_community.tools.tavily_search import TavilySearchResults
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
    raw_json = flight_search.invoke(json.dumps(kwargs))      # ← Skyscanner 原始 JSON str
    flights_list = json.loads(raw_json)["flights"]           # ← list[dict]
    return {"flights": flights_list}                         # ★ 关键：带上 'flights' 键


flight_tool = StructuredTool.from_function(
    name="flight_search",
    description="Search flights via Skyscanner",
    func=flight_search_bridge,
    args_schema=FlightSearchInput,
    return_direct=False,
)

# ----------------------------------------------------------------------
# 4.  LLM
# ----------------------------------------------------------------------
llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.2,
    openai_api_key=OPENAI_API_KEY
)

# ----------------------------------------------------------------------
# 5.  System prompt
# ----------------------------------------------------------------------
today = datetime.datetime.now().strftime("%Y-%m-%d")
SYSTEM_PROMPT = """
Today is {today}. You are an intelligent flight‑booking assistant that can
(1) understand free‑form user requests, (2) call the tool **flight_search**
to fetch real‑time flight options, and (3) call **rate_flights** to rank and
comment on those options before replying.

===================== RULES =====================
1. Use exactly the origin / destination strings provided by the user.
2. If a city is ambiguous or date missing, ASK for clarification.
3. Convert relative dates like “tomorrow” to YYYY‑MM‑DD.
4. After you have all *required* fields (origin, destination, departure_date),
   call **flight_search**.
5. When flight_search returns, IMMEDIATELY call **rate_flights** with
   `{{"flights": <flight_search result>}}`.
6. Your **final reply must be the JSON string returned by rate_flights** —
   nothing else.  This lets the UI parse it directly.
=================================================

MEMORY RULES:
• Conversation history is reliable; reuse previously given values unless the
  user changes them.
• Only ask follow‑up questions for fields still UNKNOWN.

==================== EXAMPLE ====================
User: Show me flights from Paris to Rome on 2025‑05‑10.

Assistant:
Thought: I have origin, destination, departure_date → call flight_search.
Action: flight_search
Action Input: {{"origin":"Paris","destination":"Rome","departure_date":"2025-05-10"}}

# (system executes the tool; returns)
Observation: {{"flights":[...]}}          # list[dict]

Thought: Need to rank the flights.
Action: rate_flights
Action Input: {{"flights": (flight_search result)}}

# (system executes the tool; returns)
Observation: {{"recommended":{{...}}, "flights":[...], "commentary":"..."}}

Thought: I now know the final answer.
Final Answer: (rate_flights result)       # ← this exact JSON string
================ END EXAMPLE ================

Remember: **Never** skip the rate_flights step.  
If rate_flights validation fails, diagnose the error, fix the input, and try
again — do *not* exit the conversation until rate_flights succeeds.
""".format(today=today)


# ----------------------------------------------------------------------
# 6.  Build graph WITH memory
# ----------------------------------------------------------------------

memory_store = MemorySaver()  
os.environ["TAVILY_API_KEY"] = "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"

tavily_tool   = TavilySearchResults(max_results=5)                                # ### NEW
TOOLS = [flight_tool, tavily_tool,rate_flights_tool] # we already imported the rate_flights_tool

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
