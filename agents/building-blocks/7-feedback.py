"""
Feedback: Provides strategic points where human judgement is required.
This component implements approval workflows and human-in-the-loop processes for high-risk decisions or complex judgments.
"""

from openai import OpenAI


def get_human_approval(content: str) -> bool:
    print(f"Generated content:\n{content}\n")
    response = input("Approve this? (y/n): ")
    return response.lower().startswith("y")


def intelligence_with_human_feedback(prompt: str) -> None:
    client = OpenAI()

    response = client.chat.completions.create(
        model="azure/gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    draft_response = response.choices[0].message.content

    if get_human_approval(draft_response):
        print("Final answer approved")
    else:
        print("Answer not approved")


if __name__ == "__main__":
    intelligence_with_human_feedback("Write a short poem about technology")
