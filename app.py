from flask import Flask, request, jsonify, render_template, send_from_directory
import logging
import json
import datetime
import os
import shutil

# Import the agent components from Agent_Langgraph.py
from Agent_Langgraph import graph, memory_store

# Set up logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(message)s',
    encoding='utf-8'
)

app = Flask(__name__, static_folder='static', template_folder='templates')

# Dictionary to store session thread_ids
session_threads = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

def agent_execute_with_retry(query, chat_history=None):
    """
    Execute the LangGraph agent with retry logic
    
    Args:
        query (str): User's query
        chat_history (list): Previous conversation history
        
    Returns:
        tuple: (success, result, updated_history)
    """
    max_retries = 2
    attempts = 0
    
    # Initialize message history if not provided
    if chat_history is None:
        chat_history = []
    
    # Get or create a thread_id for this session
    thread_id = f"session-{len(chat_history)}-{datetime.datetime.now().timestamp()}"
    
    while attempts < max_retries:
        try:
            # Prepare input for the agent
            input_messages = []
            
            # Add chat history to the input
            for user_msg, assistant_msg in chat_history:
                input_messages.append(("user", user_msg))
                input_messages.append(("assistant", assistant_msg))
            
            # Add the current query
            input_messages.append(("user", query))
            
            # Call the LangGraph agent
            inputs = {"messages": input_messages}
            config = {"configurable": {"thread_id": thread_id}}
            
            result = graph.invoke(inputs, config=config)
            
            # Extract the assistant's reply
            assistant_reply = result["messages"][-1].content
            
            # Update chat history
            updated_history = chat_history.copy()
            updated_history.append((query, assistant_reply))
            
            return True, assistant_reply, updated_history
            
        except Exception as e:
            logging.error(f"Error in agent execution (attempt {attempts+1}): {str(e)}")
            attempts += 1
            if attempts >= max_retries:
                return False, f"I'm sorry, I couldn't process your request after {max_retries} attempts. Please try again later.", chat_history
    
    return False, "Something went wrong. Please try again.", chat_history

@app.route('/api/search', methods=['POST'])
def search_flights():
    data = request.json
    query = data.get('query', '')
    session_id = data.get('session_id', 'default-session')
    
    # Initialize chat history for this session if it doesn't exist
    if session_id not in session_threads:
        session_threads[session_id] = []
    
    logging.info(f"Query: {query}")
    
    try:
        # Call the agent with the session's chat history
        success, result, history = agent_execute_with_retry(query, chat_history=session_threads[session_id])
        
        # Update the session's chat history
        session_threads[session_id] = history
        
        return jsonify({
            'success': success,
            'result': result,
            'session_id': session_id
        })
        
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({
            'success': False,
            'result': "I'm sorry, there was an error processing your flight search. Please try again with more specific details.",
            'session_id': session_id
        })

@app.route('/api/clear_chat', methods=['POST'])
def clear_chat():
    data = request.json
    session_id = data.get('session_id', 'default-session')
    
    # Clear the chat history for this session
    if session_id in session_threads:
        session_threads[session_id] = []
    
    logging.info(f"Cleared chat history for session: {session_id}")
    
    return jsonify({
        'success': True,
        'message': 'Chat history cleared',
        'session_id': session_id
    })

if __name__ == '__main__':
    # Create necessary directories if they don't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    if not os.path.exists('static'):
        os.makedirs('static')
        
    # Move index.html to templates folder if it exists in current directory
    if os.path.exists('index.html') and not os.path.exists('templates/index.html'):
        shutil.copy('index.html', 'templates/index.html')
    
    logging.info(f"Server starting at: {datetime.datetime.now()}")
    app.run(debug=True, port=5000)