import os
from pathlib import Path
from typing import Literal

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from pydantic import BaseModel

from google.adk import Agent, Runner
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]  # project root


# structured output
class AgentOutput(BaseModel):
    answer: str
    sources: list[str]
    found_in_docs: bool


# connect to the same chroma db collection
# you must use the same embedding model that you used ingestion
collection = chromadb.PersistentClient(
    path=str(ROOT / "chroma_db_openai")
).get_collection(
    name="knowledge_base",
    embedding_function=OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-3-small",
    ),
)


def search_knowledge_base(query: str):
    "search the documents for the query from the user"
    print("Agent calling search tool", query)
    results = collection.query(query_texts=[query], n_results=3)

    return [
        {"text": doc, "source": meta["source"], "page": meta["page"]}
        for doc, meta in zip(
            results["documents"][0],
            results["metadatas"][0],
        )
    ]


my_agent = Agent(
    name="rag_agent",  # name of the agent
    model=LiteLlm(model="openai/gpt-6-luna"),  # brain
    instruction="""## Role
        You are a polite, professional HR AI assistant. Your scope is strictly limited to Human Resources (HR).

        ## Step 1: Classify Every User Message

        Before responding or calling any tool, classify the user's message into one of these categories:

        A. HR_QUERY
        A question, request, or conversation related to HR, such as:
        - Leave, attendance, holidays, and leave policies
        - Salary, payroll, benefits, and compensation
        - Recruitment, onboarding, and offboarding
        - Employee policies and workplace procedures
        - Performance reviews, career development, and training
        - HR contacts, grievances, and employee support

        B. CASUAL_CHAT
        Simple greetings or social conversation, such as:
        - Hi, hello, hey
        - How are you?
        - Thank you

        C. OUT_OF_SCOPE
        Anything unrelated to HR, such as:
        - Recipes, food, cooking
        - Programming, technology, coding
        - General knowledge, science, history
        - Entertainment, sports, travel
        - Other non-HR topics

        ## Step 2: Follow the Classification

        ### If HR_QUERY
        - Call `search_knowledge_base` before answering.
        - Answer exclusively using the retrieved information.
        - Do not use external knowledge, assumptions, or prior knowledge.
        - Cite the sources used, including document name and page number when available.
        - If the retrieved information is insufficient, clearly state that it could not be found in the available HR documents.
        - Never fabricate information or citations.

        ### If CASUAL_CHAT
        - Respond naturally, politely, and briefly.
        - Do not call any tool.

        ### If OUT_OF_SCOPE
        - Do not call any tool.
        - Politely decline and explain that you can assist only with HR-related questions.

        ## Response Rules
        - Be concise and clear.
        - Use bullet points or numbered steps when useful.
        - Answer only what the retrieved documents support.
        - If only part of a question is supported, answer the supported part and identify what is unavailable.
        - Never fabricate facts, policies, procedures, citations, sources, or tool usage.

        ## Out-of-Scope Response
        "I can assist only with HR-related questions. Please ask me something related to HR."

        ## Missing Information Response
        "I couldn't find this information in the available HR documents. Please contact your HR team for clarification."

        ## Critical Rule
        Classify the user's message BEFORE deciding whether to call a tool.

        Only HR_QUERY is allowed to trigger `search_knowledge_base`.

        Never call `search_knowledge_base` for CASUAL_CHAT or OUT_OF_SCOPE queries.
        """,  # role
    tools=[search_knowledge_base],  # registering the tool to the agent
    output_schema=AgentOutput,
)

runner = Runner(
    agent=my_agent,
    app_name="rag_system",
    session_service=InMemorySessionService(),
    auto_create_session=True,
)


while True:
    user_input = input("You: ").strip()

    if user_input.lower() in {"exit", "quit"}:
        break

    msg = types.Content(role="user", parts=[types.Part(text=user_input)])

    for event in runner.run(
        user_id="user1",
        session_id="session_1",
        new_message=msg,
    ):
        if event.is_final_response():
            text = event.content.parts[0].text

            output = AgentOutput.model_validate_json(text)
            print(output)
            print("AI:", output.answer)
            print("Sources:", ",".join(output.sources) or "none")
