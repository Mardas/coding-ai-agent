import os
from dotenv import load_dotenv
from openai import OpenAI
import langchain
import json
from liftmath import (
    estimate_one_rm,                                       # six-formula 1RM consensus
    load_plates, load_plates_from_inventory,               # plate loading (unlimited / finite)
    score,                                                 # all four scores at once
    wilks_score, wilks_original_score,                     # Wilks 2020 / original
    dots_score, ipf_gl_points,                             # DOTS / IPF GL points
    search_records, weight_class_for, percent_of_record,   # record lookup
    parse_mark, format_seconds,                            # "4:12.3" <-> seconds
    records_as_of,                                         # dataset snapshot date
    lbs_to_kg, kg_to_lbs, convert_weight,                  # exact lb/kg conversion
    parse_strong_csv, parse_hevy_csv,                      # Strong/Hevy CSV export -> WorkoutSet
    e1rm_trend, weekly_tonnage,                            # derived views over imported history
    to_dict, to_json,                                      # serialize any result
)

load_dotenv()


plates = load_plates(155, unit="kg")
print(plates.plates) 
def load_plates_tool(weight: float, unit: str = "kg") -> str:
    plates = load_plates(weight, unit=unit)
    result = { "plates": []}
    for pl in plates.plates:
        result["plates"].append({
            "weight": pl[0],
            "count": pl[1]
        })
    return json.dumps(result.get("plates"))
# result = load_plates_tool(155, unit="kg")
# print(result)
TOOL_SHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "load_plates_tool",
            "description": "Which plates and how many of them are needed to load a given weight for each side of the bar. ",
            "parameters": {
                "type": "object",
                "properties": {
                    "weight": {
                        "type": "number",
                        "description": "The target bar weight to load"
                    },
                    "unit": {
                        "type": "string",
                        "description": "Weight unit",
                        "enum": ["kg", "lbs"]
                    }
                },
                "required": ["weight", "unit"]
            }
        }
    }
]
TOOLS = {
    "load_plates_tool": load_plates_tool
    }
openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

messages = []

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ("exit", "quit", "bye"):
        break

    messages.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOL_SHEMAS
    )
    message = response.choices[0].message
    messages.append(message)

    if not message.tool_calls:
        print(message.content)
        break

    for tool_call in message.tool_calls:
        args = json.loads(tool_call.function.arguments)
        print(f"Model wants to run {tool_call.function.name} with args: {args}")

        func = TOOLS.get(tool_call.function.name)
        if func is None:
            result = f"Error: unknown tool '{tool_call.function.name}'"
        else:
            try:
                result = func(**args)
            except Exception as e:
                result = f"Error: {e}"
        
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })
    print("--------------------------------")