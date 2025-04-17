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
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
)

# ----------------------------------------------------------------------
# 5.  Build the agent – OPENAI_FUNCTIONS uses the function‑calling protocol
#     system_message sets the “persona” you passed into your old llm() wrapper
# ----------------------------------------------------------------------
today = datetime.datetime.now().strftime("%Y-%m-%d")
system_prompt = (
    "You are a helpful assistant specialized in flight booking. "
    f"Today is {today}. You are a helpful flight‑booking assistant. "
    "When the user gives relative dates such as 'tomorrow', 'next Friday', "
    "or 'in two weeks', interpret them relative to today. When interpreting relative dates, always convert them to an ISO date string (YYYY-MM-DD) before inserting them into function arguments. "
    
    "When the user asks for flights, collect the required details and call the "
    "`flight_search` function. If any required detail is missing, ask ONLY for "
    "those fields. Present final results clearly in English."
)



agent = initialize_agent(
    tools=[flight_tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    verbose=True,  # prints each step for debugging; set False in prod
    agent_kwargs={"system_message": system_prompt},
)

# ----------------------------------------------------------------------
# 6.  Example use
# ----------------------------------------------------------------------
def chat_loop():
    print(f"🛫 Flight Search AI Agent 🛬 | Type 'exit' to quit.")
    while True:
        user_msg = input("You: ").strip()
        if user_msg.lower() in {"exit", "quit"}:
            break
        bot_reply = agent.invoke(user_msg)   # <-- 完整上下文由 memory 保存
        print(f"Bot: {bot_reply}\n")

if __name__ == "__main__":
    chat_loop()


    # print(agent.run(
    #     "SFO → NRT, 1 adult,economy."
    # ))



