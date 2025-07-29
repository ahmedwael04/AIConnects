from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from tools.serpapi_tool import activity_search_tool
from tools.fallback_tool import fallback_activity_tool
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# Updated ReAct prompt template
activity_prompt = PromptTemplate(
    input_variables=["input"],
    template="""You are an activity recommendation assistant for Egypt. Use the provided tools to find activities based on the user's location: {input}"
)

Available tools:
{tools}

Use the following tool names when invoking tools: {tool_names}

Answer the following query: {input}

To answer, think step-by-step, use the tools when necessary, and provide a clear, concise response. Record your intermediate steps in the scratchpad.

{agent_scratchpad}"""
)

activity_agent = create_react_agent(
    llm=llm,
    tools=[activity_search_tool, fallback_activity_tool],
    prompt=activity_prompt
)

activity_executor = AgentExecutor(agent=activity_agent, tools=[activity_search_tool, fallback_activity_tool], verbose=True)