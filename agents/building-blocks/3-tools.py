"""
Tools: Enables agents to execute specific actions in external systems.
This component provides the capability to make API calls, database updates, file operations, and other practical actions.


More info: https://platform.openai.com/docs/guides/function-calling?api-mode=responses
"""

import json
import requests
from openai import OpenAI


def get_weather(latitude, longitude):
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]["temperature_2m"]


def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)
    raise ValueError(f"Unknown function: {name}")


def intelligence_with_tools(prompt: str) -> str:
    client = OpenAI()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current temperature for provided coordinates in celsius.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "number"},
                        "longitude": {"type": "number"},
                    },
                    "required": ["latitude", "longitude"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        }
    ]

    input_messages = [{"role": "user", "content": prompt}]

    # Step 1: Call model with tools
    response = client.chat.completions.create(
        model="azure/gpt-4o",
        messages=input_messages,
        tools=tools,
    )

    # Check if there are tool calls
    if not response.choices[0].message.tool_calls:
        return response.choices[0].message.content

    # Step 2: Handle function calls
    for tool_call in response.choices[0].message.tool_calls:
        if tool_call.type == "function":
            # Step 3: Execute function
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            result = call_function(name, args)

            # Step 4: Append function call and result to messages
            input_messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call]
            })
            input_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,  # Updated field name
                "content": str(result),
            })

    # Step 5: Get final response with function results
    final_response = client.chat.completions.create(
        model="azure/gpt-4o",
        messages=input_messages,
        tools=tools,
    )

    return final_response.choices[0].message.content


if __name__ == "__main__":
    result = intelligence_with_tools(prompt="What's the weather like in Paris today?")
    print("Tool Calling Output:")
    print(result)
