import os
import requests
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
# * the desciption should not be same for any tools
def get_weather(latitude: str, longitude: str):
    "get the weather for a given city"
    print("agent calling weather tool")
    print(latitude, longitude)

    # Get weather
    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
            "timezone": "auto",
        },
    ).json()

    return weather["current"]


def get_products():
    "get the products"
    print("agent calling products tool")

    # Get weather
    products = requests.get("https://fakestoreapi.com/products").json()

    return products


my_agent = Agent(
    name="customer_support_agent",  # name of the agent
    model=LiteLlm(model="openai/gpt-6-luna"),  # brain
    instruction="""You are a helpful assistant.
		You must answer the questions related to weather and products only. 
        If the city is invalid tell the user appropriately before calling the tool. 
        If the question is related to products use the tool
        """,  # role
    tools=[get_weather, get_products],  # registering the tool to the agent
)

runner = Runner(
    agent=my_agent,
    app_name="app",
    session_service=InMemorySessionService(),
    auto_create_session=True,
)

msg = types.Content(role="user", parts=[types.Part(text="tell me about product id 10")])
for event in runner.run(user_id="user1", session_id="session_1", new_message=msg):
    if event.is_final_response():
        print(event.content.parts[0].text)


# uv run src/example4.py
