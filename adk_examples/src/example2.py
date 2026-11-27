import os
from dotenv import load_dotenv
from google.adk import Agent, Runner
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()


# here comes the tool
### Tools
# * tools are functions
# * the functions must have name
# * the function must have a description
# * the function must return data
# * the function can have arguments
# * the desciption should not be same for any other tools
def get_weather(city: str):
    "get the weather for a given city"
    print("agent calling weather tool")
    print(city)
    return "weather is 20 deg"


my_agent = Agent(
    name="weather_agent",  # name of the agent
    model=LiteLlm(model="openai/gpt-6-luna"),  # brain
    instruction="""You are a weather agent.
		You must answer the questions related to weather only. 
        If the city is invalid tell the user appropriately before calling the tool""",  # role
    tools=[get_weather],  # registering the tool to the agent
)

runner = Runner(
    agent=my_agent,
    app_name="app",
    session_service=InMemorySessionService(),
    auto_create_session=True,
)

msg = types.Content(role="user", parts=[types.Part(text="temperature in Pune")])
for event in runner.run(user_id="user1", session_id="session_1", new_message=msg):
    if event.is_final_response():
        print(event.content.parts[0].text)


# uv run src/example2.py
