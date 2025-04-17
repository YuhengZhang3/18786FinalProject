from flask import Flask, request, jsonify, render_template, send_from_directory
import logging
import json
import datetime
from agent import agent_execute_with_retry

# Set up logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(message)s',
    encoding='utf-8'
)

app = Flask(__name__, static_folder='static')

# Dictionary to store chat histories by session ID
chat_histories = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/api/search', methods=['POST'])
def search_flights():
    data = request.json
    query = data.get('query', '')
    session_id = data.get('session_id', 'default-session')
    
    # Initialize chat history for this session if it doesn't exist
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    logging.info(f"Query: {query}")
    
    try:
        # Call the agent with the session's chat history
        success, result, history = agent_execute_with_retry(query, chat_history=chat_histories[session_id])
        
        # Update the session's chat history
        chat_histories[session_id] = history
        
        return jsonify({
            'success': True,
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
    if session_id in chat_histories:
        chat_histories[session_id] = []
    
    logging.info(f"Cleared chat history for session: {session_id}")
    
    return jsonify({
        'success': True,
        'message': 'Chat history cleared',
        'session_id': session_id
    })

if __name__ == '__main__':
    logging.info(f"Server starting at: {datetime.datetime.now()}")
    app.run(debug=True, port=5000)