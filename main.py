from langchain_core.messages import HumanMessage #allows building AI apps
from langchain_openai import ChatOpenAI #allows use openai
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent #allows building AI agents
from dotenv import load_dotenv #loads env files

load_dotenv()

@tool
def calculator(a: float, b: float) -> str:
    """Useful for performing basic addition arithmetic calculations with numbers"""
    return f"The sum of {a} and {b} is {a + b}."

def main():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    tools = [calculator]
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