import datetime
from agent import agent_execute_with_retry

def main():
    current_date = datetime.datetime.now()
    print(f"🛫 Flight Search AI Agent 🛬")
    print(f"----------------------------")
    print(f"Current system date: {current_date}")
    
    print("Ask me anything about flights, or type 'exit' to quit.")
    print("Example: 'I need to find a business class flight from SFO to ORD next Friday for 2 adults'")
    
    my_history = []
    while True:
        query = input('\n✈️ Query: ')
        if query.lower() in ['exit', 'quit', 'bye']:
            print("Thank you for using Flight Search AI Agent. Goodbye!")
            break
        else:
            success, result, my_history = agent_execute_with_retry(query, chat_history=my_history)
            my_history = my_history[-5:]  # Keep only last 5 interactions for context
            print(f"\n{result}")

if __name__ == "__main__":
    main()
    
# one-way
# I need to find a business class flight from SFO to ORD next Friday for 2 adults
# round-trip
# Find me a round-trip flight from New York to Los Angeles, leaving next Friday and returning next Sunday, economy class.
