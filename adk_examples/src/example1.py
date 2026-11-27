import os
from dotenv import load_dotenv
from google.adk import Agent, Runner
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()

# This wont answer about weather in specific cities. there are no tools to the agent
# check example 2 .

my_agent = Agent(
    name="weather_agent",  # name of the agent
    model=LiteLlm(model="openai/gpt-6-luna"),  # brain
    instruction="You are a weather agent. You should answer the questions related to weather",  # role
)

runner = Runner(
    agent=my_agent,
    app_name="app",
    session_service=InMemorySessionService(),
    auto_create_session=True,
)

msg = types.Content(role="user", parts=[types.Part(text="Tell me the weather in SF")])
for event in runner.run(user_id="user1", session_id="session_1", new_message=msg):
    if event.is_final_response():
        print(event.content.parts[0].text)


# uv run src/example1.py
