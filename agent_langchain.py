import json
import datetime
from llm import llm
from flight_search import SkyscannerFlightSearchTool
from nlp import extract_flight_parameters
import datetime


# pip install langchain==0.2.0 openai pydantic

import datetime
import os
from pydantic import BaseModel, Field, validator
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

from config import OPENAI_API_KEY
from flight_search import SkyscannerFlightSearchTool   # your existing class

# ----------------------------------------------------------------------
# 1.  Keep your original flight search instance
# ----------------------------------------------------------------------
flight_search = SkyscannerFlightSearchTool()

# ----------------------------------------------------------------------
# 2.  Define a strict input schema (required fields with ..., optionals with defaults)
#     Only these REQUIRED fields will trigger follow‑up questions.
# ----------------------------------------------------------------------
class FlightSearchInput(BaseModel):
    origin: str = Field(..., description="IATA code or city name of the departure location")
    destination: str = Field(..., description="IATA code or city name of the arrival location")
    departure_date: str = Field(..., description="YYYY-MM-DD")
    # optional ↓ — the model will not chase the user for them unless it wants to
    return_date: str | None = Field(None, description="YYYY-MM-DD if round‑trip")
    adults: int = Field(1, ge=1, le=9)
    cabin_class: str = Field(
        "Economy",
        description="Economy | Premium Economy | Business | First"
    )
    children: int = 0
    infants: int = 0
    


    # → optional: basic date sanity check (convert 6/12 → 2025-06-12 yourself if you like)
    @validator("departure_date", "return_date", pre=True, always=True)
    def _check_date_fmt(cls, v):
        if v is None:
            return v
        try:
            datetime.datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("date must be YYYY-MM-DD")

# ----------------------------------------------------------------------
# 3.  Wrap your existing invoke() in a LangChain StructuredTool
# ----------------------------------------------------------------------
def flight_search_bridge(**kwargs):
    return flight_search.invoke(json.dumps(kwargs))   # kwargs ← 已校验 dict

flight_tool = StructuredTool.from_function(
    name="flight_search",
    description="Search flights via Skyscanner",
    func=flight_search_bridge,
    args_schema=FlightSearchInput,
    return_direct=True
)

# ----------------------------------------------------------------------
# 4.  Create the LLM (ChatOpenAI uses the same openai‑python backend you already call)
# ----------------------------------------------------------------------
llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
)

# ----------------------------------------------------------------------
# 5.  Build the agent – OPENAI_FUNCTIONS uses the function‑calling protocol
#     system_message sets the “persona” you passed into your old llm() wrapper
# ----------------------------------------------------------------------
today = datetime.datetime.now().strftime("%Y-%m-%d")
system_prompt = f"""
Today is {today}. You are a flight‑booking assistant.

RULES:
1. Use **exactly** the origin and destination strings the user provides.
   - If the user writes "Pittsburgh", set origin="Pittsburgh" (or the PIT IATA code) – never replace it with another city.
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




# 1‑次性初始化
agent = initialize_agent(
    tools=[flight_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    ),
    verbose=False,
    agent_kwargs={"system_message": system_prompt},
)

def chat_loop():
    print("🛫 Flight Search AI Agent 🛬 | Type 'exit' to quit.")
    while True:
        user_msg = input("You: ").strip()
        if user_msg.lower() in {"exit", "quit"}:
            break
        bot_reply = agent.invoke(user_msg)    # ✅ 每次用同一个 agent
        print("Bot:", bot_reply, "\n")


if __name__ == "__main__":
    chat_loop()


    # print(agent.run(
    #     "SFO → NRT, 1 adult,economy."
    # ))



