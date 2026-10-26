from langchain_core.messages import HumanMessage #allows building AI apps
from langchain_openai import ChatOpenAI #allows use openai
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent #allows building AI agents
from dotenv import load_dotenv #loads env files
from datetime import datetime #python date/time library
import requests #allows uv to talk to websites/APIs

load_dotenv()

@tool
def currentTime() -> str:
    """Get the current date and time. Useful when asked what time or day it is today"""
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")

def main():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    tools = [currentTime]
    agent_executor = create_react_agent(model, tools)

    print("\nHi! I'm your AI assistant.")
    print("You can chat with me, or ask me to perform calculations.")
    print("Type 'quit' to exit.")

    messages = []

    while True:
        user_input = input("\nYou: ").strip()
        if user_input == "quit":
            break

        messages.append(HumanMessage(content=user_input))

        print("\nAssistant: ", end="")
        for chunk in agent_executor.stream({"messages": messages}):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="")
                    messages.append(message)
            if "tools" in chunk and "messages" in chunk["tools"]:
                for message in chunk["tools"]["messages"]:
                    messages.append(message)
        print()

if __name__ == "__main__":
    main()