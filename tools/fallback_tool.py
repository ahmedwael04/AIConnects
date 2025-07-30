import json
from langchain.tools import Tool

def load_fallback_data(location, category):
    with open("data/fallback_data.json", "r") as f:
        data = json.load(f)
    results = data.get(location, {}).get(category, [])
    print(f"Fallback Tool - Location: {location}, Category: {category}, Results: {results}")
    return results

fallback_food_tool = Tool(
    name="FallbackFood",
    func=lambda x: load_fallback_data(x.split("|")[0], "food"),
    description="Fetches food recommendations from local JSON database."
)

fallback_activity_tool = Tool(
    name="FallbackActivity",
    func=lambda x: load_fallback_data(x.split("|")[0], "activities"),
    description="Fetches activity recommendations from local JSON database."
)