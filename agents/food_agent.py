from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from tools.serpapi_tool import food_search_tool
from tools.fallback_tool import fallback_food_tool
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# Updated ReAct prompt template
food_prompt = PromptTemplate(
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
    template="""You are a food recommendation assistant for Egypt. Your task is to find restaurants based on the user's location and budget using the provided tools.

Available tools:
{tools}

Use the following tool names when invoking tools: {tool_names}

Answer the following query: {input}

To answer, think step-by-step, use the tools when necessary, and provide a clear, concise response. Record your intermediate steps in the scratchpad.

{agent_scratchpad}"""
)

food_agent = create_react_agent(
    llm=llm,
    tools=[food_search_tool, fallback_food_tool],
    prompt=food_prompt
)

food_executor = AgentExecutor(agent=food_agent, tools=[food_search_tool, fallback_food_tool], verbose=True)