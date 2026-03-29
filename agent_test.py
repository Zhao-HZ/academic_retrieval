from llm.react_agent import create_agent__

def test():
    """Test function for multi-turn agent conversation"""
    agent = create_agent__()

    # Multi-turn conversation with memory
    print("=== Starting Multi-turn Agent Conversation ===\n")

    # Use a thread_id for conversation memory
    config = {"configurable": {"thread_id": "test_session"}}

    # First turn: Ask about knowledge base
    print("User: What can you help me with?")
    response = agent.invoke(
        {"messages": [("user", "What can you help me with?")]},
        config
    )
    print(f"Agent: {response['messages'][-1].content}\n")

    # Second turn: Test retrieval
    print("User: Can you retrieve information about machine learning?")
    response = agent.invoke(
        {"messages": [("user", "Retrieve information about machine learning from the knowledge base")]},
        config
    )
    print(f"Agent: {response['messages'][-1].content}\n")

    # Third turn: Test calculator
    print("User: What's the square root of 144?")
    response = agent.invoke(
        {"messages": [("user", "Calculate the square root of 144")]},
        config
    )
    print(f"Agent: {response['messages'][-1].content}\n")

    # Fourth turn: Test current time
    print("User: What time is it now?")
    response = agent.invoke(
        {"messages": [("user", "What time is it now?")]},
        config
    )
    print(f"Agent: {response['messages'][-1].content}\n")

    print("=== Conversation Ended ===")
    
if __name__ == "__main__":
    test()
    