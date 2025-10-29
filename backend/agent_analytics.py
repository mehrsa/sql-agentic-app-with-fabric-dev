import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from sqlalchemy.pool import QueuePool

from chat_data_model import init_chat_db
from shared.connection_manager import sqlalchemy_connection_creator, connection_manager

load_dotenv(override=True)

app = Flask(__name__)
CORS(app)

# Database configuration for Fabric SQL (analytics data)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = "mssql+pyodbc://"
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'creator': sqlalchemy_connection_creator,
    'poolclass': QueuePool,
    'pool_size': 5,
    'max_overflow': 10,
    'pool_pre_ping': True,
    'pool_recycle': 3600,
    'pool_reset_on_return': 'rollback'
}

db = SQLAlchemy(app)

# Initialize chat history module with database
init_chat_db(db)
from chat_data_model import (
    ToolDefinition, ChatHistoryManager,
    handle_chat_sessions, 
    clear_chat_history, clear_session_data, initialize_tool_definitions, 
    initialize_agent_definitions
)

# Chat History API Routes
@app.route('/api/chat/sessions', methods=['GET', 'POST'])
def chat_sessions_route():
    print("Handling chat sessions request...")
    return handle_chat_sessions(request)

@app.route('/api/chat/history/<session_id>', methods=['GET'])
def get_chat_history(session_id):
    """New endpoint to retrieve chat history for a session."""
    try:
        # Use the ChatHistoryManager to get the history
        chat_manager = ChatHistoryManager(session_id=session_id)
        history = chat_manager.get_conversation_history()
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/clear-chat-history', methods=['DELETE'])
def clear_chat_route():
    return clear_chat_history()

@app.route('/api/admin/clear-session/<session_id>', methods=['DELETE'])
def clear_session_route(session_id):
    return clear_session_data(session_id)

@app.route('/api/tools/definitions', methods=['GET', 'POST'])
def handle_tool_definitions():
    if request.method == 'GET':
        tools = ToolDefinition.query.filter_by(is_active=True).all()
        return jsonify([tool.to_dict() for tool in tools])
    
    if request.method == 'POST':
        data = request.json
        tool_def = ToolDefinition(
            name=data['name'],
            description=data.get('description'),
            input_schema=data['input_schema'],
            version=data.get('version', '1.0.0'),
            cost_per_call_cents=data.get('cost_per_call_cents', 0)
        )
        db.session.add(tool_def)
        db.session.commit()
        return jsonify(tool_def.to_dict()), 201

# Endpoints for logging messages from banking service
@app.route('/api/chat/log-trace', methods=['POST'])
def log_trace():
    import traceback

    try:
        data = request.json
        chat_manager = ChatHistoryManager(
            session_id=data.get('session_id'),
            user_id=data.get('user_id')
        )
        # call into the chat history manager (note: method expects 'message' kw)
        chat_manager.add_trace_messages(
            serialized_messages=data.get('messages'),
            trace_duration=data.get('trace_duration')
        )

        return jsonify({"status": "success"}), 201

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    
def initialize_analytics_app():
    """Initialize analytics app when called from combined launcher."""
    with app.app_context():
        db.create_all()
        initialize_tool_definitions()
        initialize_agent_definitions()
        print("[Analytics Service] Database tables initialized")