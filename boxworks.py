import os
from dotenv import load_dotenv

from langchain.agents import AgentExecutor, create_openai_tools_agent

from langchain_box.retrievers import BoxRetriever
from langchain_box.utilities import BoxAuth, BoxAuthType

from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import ChatOpenAI

from langchain.tools.retriever import create_retriever_tool


load_dotenv(".env")

box_client_id=os.getenv("BOX_CLIENT_ID")
box_client_secret=os.getenv("BOX_CLIENT_SECRET")
box_user_id=os.getenv("BOX_USER_ID")
box_file_ids=[os.getenv("BOX_FIRST_FILE")]

openai_key = os.getenv("OPENAI_API_KEY")

prompt="Provide a three paragraph description of the character Victor"

auth = BoxAuth(
    auth_type=BoxAuthType.CCG,
    box_client_id=box_client_id,
    box_client_secret=box_client_secret,
    box_user_id=box_user_id,
)

retriever = BoxRetriever( 
    box_auth=auth,
    box_file_ids=box_file_ids,
    citations=True
)

box_ai_tool = create_retriever_tool(
    retriever,
    "box_ai_tool",
    "This tool is used to use Box AI to process a file and retrieve the response and citations from Box AI"
)
tools = [box_ai_tool]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
prompt.messages

llm = ChatOpenAI(temperature=0)

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

for chunk in agent_executor.stream(
    { "input": "With Box AI, retrieve a 5 paragraph description of the character Victor. From that response, recommend five actors to play the role of Victor" }
):
    print(chunk)
    print("----")

print(f"\n\n{chunk['output']}")