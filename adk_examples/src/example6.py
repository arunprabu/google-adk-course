import os
import requests
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal
from google.adk import Agent, Runner
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()


# structured output
class AgentOutput(BaseModel):
    response_count: int
    role: Literal["ai", "user"]
    response: str
    status: bool
    topic: str


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
    instruction="""Assistant Instructions

        You are a helpful, polite AI assistant restricted to two topics: Weather and Products.

        Weather: Answer weather-related questions using the weather tool. Validate explicitly provided city names before calling the tool. If a city is invalid or unclear, ask the user for a valid location. Never fabricate weather information.
        Products: Always use the product search tool for product-related queries. Provide relevant recommendations, comparisons, specifications, and prices when available. Never invent product details.
        Unsupported Topics: Politely decline all requests unrelated to weather or products.
        Briefly explain your supported topics.
        Gender Neutrality: Never assume or associate the user with any gender. Always use gender-neutral language.
        Response Style: Keep answers concise, clear, accurate, and friendly. Ask for clarification only when necessary.

        Core Rule: Stay strictly within Weather and Products. Never fabricate information or claim tool usage without actually using the tool.
        """,  # role
    tools=[get_weather, get_products],  # registering the tool to the agent
    output_schema=AgentOutput,
)

runner = Runner(
    agent=my_agent,
    app_name="app",
    session_service=InMemorySessionService(),
    auto_create_session=True,
)


while True:
    user_input = input("You: ").strip()
    if user_input.lower() in {"exit", "quit"}:
        break
    msg = types.Content(role="user", parts=[types.Part(text=user_input)])
    for event in runner.run(user_id="user1", session_id="session_1", new_message=msg):
        if event.is_final_response():
            output = AgentOutput.model_validate_json(event.content.parts[0].text)
            print(output)

            # print("AI:", event.content.parts[0].text)


# uv run src/example5.py
