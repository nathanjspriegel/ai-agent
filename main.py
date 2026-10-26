from langchain_core.messages import HumanMessage #allows building AI apps
from langchain_openai import ChatOpenAI #allows use openai
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent #allows building AI agents
from dotenv import load_dotenv #loads env files

load_dotenv()

def main():
    model = ChatOpenAI(temperature=0)

    tools = []
    agent_executor = create_react_agent(model, tools)

    print("Hi! I'm your AI assistant.")
    print("You can chat with me, or ask me to perform calculations.")
    print("Type 'quit' to exit.")

    while True:
        user_input = input("\n: ").strip()

        if user_input == "quit":
            break

        print("\nAssistant: ", end="")
        for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]}
        ):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content, end="")
        print()

if __name__ == "__main__":
    main()