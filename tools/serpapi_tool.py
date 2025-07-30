from langchain.tools import Tool
from serpapi import GoogleSearch
import os
from dotenv import load_dotenv

load_dotenv()

def search_web(query, location):
    params = {
        "q": query,
        "location": location,
        "api_key": os.getenv("SERPAPI_API_KEY")
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results.get("organic_results", [])[:3]  # Return top 3 results
    print(f"SerpAPI - Query: {query}, Location: {location}, Results: {organic_results}")
    return organic_results

food_search_tool = Tool(
    name="FoodSearch",
    func=lambda x: search_web(f"best restaurants in {x}", x.split("|")[0]),
    description="Searches for restaurants based on location and budget."
)

activity_search_tool = Tool(
    name="ActivitySearch",
    func=lambda x: search_web(f"popular activities in {x}", x.split("|")[0]),
    description="Searches for activities based on location."
)