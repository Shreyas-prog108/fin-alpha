import asyncio
import sys
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from agents.agent import FinAgent
else:
    from .agent import FinAgent
async def main():
    agent = FinAgent()
    print("🤖 FinAgent - Financial Analysis Agent")
    print("=" * 50)
    print("Ask me anything about stocks!")
    print("Type 'quit' to exit\n")
    while True:
        query = input("You: ")
        if query.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if not query.strip():
            continue
        print("\n🔄 Analyzing...\n")
        try:
            result = await agent.run(query)
            response = result["messages"][-1]["content"]
            print(f"Agent: {response}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")
if __name__ == "__main__":
    asyncio.run(main())