"""
Intelligence: The "brain" that processes information and makes decisions using LLMs.
This component handles context understanding, instruction following, and response generation.

More info: https://platform.openai.com/docs/guides/text?api-mode=responses
"""

from openai import OpenAI


def basic_intelligence(prompt: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="azure/gpt-4o",  # Model configured in your LiteLLM proxy
        messages=[{"role": "user", "content": prompt}]
    )
    # print(response)
    return response.choices[0].message.content

    # response = client.responses.create(model="azure/gpt-4o", input=prompt)
    # return response.output_text


if __name__ == "__main__":
    result = basic_intelligence(prompt="What is artificial intelligence?")
    print("Basic Intelligence Output:")
    print(result)
