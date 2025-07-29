from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

router_prompt = PromptTemplate(
    input_variables=["query"],
    template="Classify the query as 'food' or 'activity': {query}"
)

router_chain = LLMChain(llm=llm, prompt=router_prompt)

def route_query(query):
    result = router_chain.run(query=query)
    return "food" if "food" in result.lower() else "activity"