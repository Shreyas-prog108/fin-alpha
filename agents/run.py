import asyncio
import sys
import re
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from agents.agent import FinAgent
else:
    from .agent import FinAgent

# Common greetings and casual phrases that shouldn't trigger analysis
GREETINGS = {
    "hi", "hii", "hiii", "hiiii", "hey", "hello", "hola", "howdy",
    "good morning", "good afternoon", "good evening", "good night",
    "morning", "afternoon", "evening", "sup", "yo", "whats up",
    "what's up", "wassup", "how are you", "how r u", "how are u",
    "thanks", "thank you", "thank u", "thx", "bye", "goodbye",
    "see you", "see ya", "later", "cya", "ok", "okay", "cool",
    "nice", "great", "awesome", "test", "testing", "help"
}

GREETING_RESPONSES = {
    "hi": "👋 Hi there! I'm FinAgent, your financial analysis assistant. Ask me about any stock - for example:\n• \"What's the price of Apple?\"\n• \"Analyze Tesla for investment\"\n• \"Give me risk analysis for HDFC Bank\"",
    "help": "🤖 **FinAgent Help**\n\nI can help you with:\n• **Stock prices**: \"What's the price of Reliance?\"\n• **Investment analysis**: \"Should I invest in TCS?\"\n• **Risk assessment**: \"Risk analysis for Infosys\"\n• **Sentiment**: \"What's the sentiment on HDFC Bank?\"\n• **News**: \"Latest news about Tesla\"\n\nJust type your question!",
    "default": "👋 Hello! How can I help you with your financial analysis today? Ask me about any stock!"
}

def is_greeting(text: str) -> bool:
    """Check if the input is a casual greeting or non-financial query"""
    cleaned = text.lower().strip().rstrip('!?.,:;')
    cleaned = re.sub(r'[^\w\s]', '', cleaned)
    return cleaned in GREETINGS or len(cleaned) <= 3

def get_greeting_response(text: str) -> str:
    """Get appropriate response for greeting"""
    cleaned = text.lower().strip().rstrip('!?.,:;')
    cleaned = re.sub(r'[^\w\s]', '', cleaned)
    if cleaned in ["help", "?"]:
        return GREETING_RESPONSES["help"]
    if cleaned.startswith("hi") or cleaned in ["hey", "hello", "hola", "howdy", "yo", "sup"]:
        return GREETING_RESPONSES["hi"]
    return GREETING_RESPONSES["default"]

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
        
        # Check for greetings first
        if is_greeting(query):
            print(f"\nAgent: {get_greeting_response(query)}\n")
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