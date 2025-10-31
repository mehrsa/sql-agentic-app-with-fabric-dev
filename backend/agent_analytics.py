import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from sqlalchemy.pool import QueuePool

from chat_data_model import init_chat_db
from shared.connection_manager import sqlalchemy_connection_creator

load_dotenv(override=True)

app = Flask(__name__)
CORS(app)

# Database configuration for Fabric SQL (analytics data)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = "mssql+pyodbc://"
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'creator': sqlalchemy_connection_creator,
    'poolclass': QueuePool,
    'pool_size': 10,
    'max_overflow': 20,
    'pool_pre_ping': True,
    'pool_recycle': 3600,
    'pool_reset_on_return': 'rollback'
}

db = SQLAlchemy(app)

# Initialize chat history module with database
init_chat_db(db)
from chat_data_model import (
    ToolDefinition, ChatHistoryManager, AgentDefinition, AgentTrace,
    handle_chat_sessions, 
    clear_chat_history, clear_session_data, initialize_tool_definitions, 
    initialize_agent_definitions
)

# Multi-Agent Analytics API Routes

@app.route('/api/chat/sessions', methods=['GET', 'POST'])
def chat_sessions_route():
    """Handle chat sessions with multi-agent support"""
    print("Handling multi-agent chat sessions request...")
    return handle_chat_sessions(request)

@app.route('/api/chat/history/<session_id>', methods=['GET'])
def get_chat_history(session_id):
    """Retrieve chat history for a session with multi-agent context"""
    try:
        chat_manager = ChatHistoryManager(session_id=session_id)
        history = chat_manager.get_conversation_history()
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/agent-stats/<session_id>', methods=['GET'])
def get_session_agent_stats(session_id):
    """Get agent usage statistics for a specific session"""
    try:
        chat_manager = ChatHistoryManager(session_id=session_id)
        stats = chat_manager.get_agent_usage_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# @app.route('/api/analytics/agent-performance', methods=['GET'])
# def get_agent_performance():
#     """Get overall agent performance analytics"""
#     try:
#         # Agent usage frequency
#         agent_usage = db.session.query(
#             db.func.coalesce(chat_data_model.ChatHistory.agent_type, 'unknown').label('agent_type'),
#             db.func.count(chat_data_model.ChatHistory.message_id).label('total_messages'),
#             db.func.sum(chat_data_model.ChatHistory.total_tokens).label('total_tokens'),
#             db.func.avg(chat_data_model.ChatHistory.response_time_ms).label('avg_response_time')
#         ).filter(
#             chat_data_model.ChatHistory.message_type == 'ai'
#         ).group_by(chat_data_model.ChatHistory.agent_type).all()
        
#         # Tool usage by agent type
#         tool_usage = db.session.query(
#             chat_data_model.ToolUsage.agent_type,
#             chat_data_model.ToolUsage.tool_name,
#             db.func.count(chat_data_model.ToolUsage.tool_call_id).label('usage_count'),
#             db.func.avg(chat_data_model.ToolUsage.execution_time_ms).label('avg_execution_time')
#         ).group_by(
#             chat_data_model.ToolUsage.agent_type,
#             chat_data_model.ToolUsage.tool_name
#         ).all()
        
#         # Session distribution
#         session_stats = db.session.query(
#             chat_data_model.ChatSession.primary_agent_type,
#             db.func.count(chat_data_model.ChatSession.session_id).label('session_count'),
#             db.func.avg(chat_data_model.ChatSession.total_agents_used).label('avg_agents_per_session')
#         ).group_by(chat_data_model.ChatSession.primary_agent_type).all()
        
#         return jsonify({
#             "agent_usage": [
#                 {
#                     "agent_type": stat[0],
#                     "total_messages": stat[1],
#                     "total_tokens": stat[2] or 0,
#                     "avg_response_time": round(stat[3] or 0, 2)
#                 } for stat in agent_usage
#             ],
#             "tool_usage": [
#                 {
#                     "agent_type": stat[0],
#                     "tool_name": stat[1],
#                     "usage_count": stat[2],
#                     "avg_execution_time": round(stat[3] or 0, 2)
#                 } for stat in tool_usage
#             ],
#             "session_stats": [
#                 {
#                     "primary_agent_type": stat[0],
#                     "session_count": stat[1],
#                     "avg_agents_per_session": round(stat[2] or 0, 2)
#                 } for stat in session_stats
#             ]
#         })
        
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

@app.route('/api/analytics/routing-analytics', methods=['GET'])
def get_routing_analytics():
    """Get analytics about agent routing decisions"""
    try:
        # Routing success rate
        routing_stats = db.session.query(
            AgentTrace.target_agent,
            AgentTrace.task_type,
            db.func.count(AgentTrace.trace_step_id).label('total_routes'),
            db.func.sum(db.case([(AgentTrace.success == True, 1)], else_=0)).label('successful_routes'),
            db.func.avg(AgentTrace.execution_duration_ms).label('avg_execution_time')
        ).group_by(AgentTrace.target_agent, AgentTrace.task_type).all()
        
        # Task type distribution
        task_distribution = db.session.query(
            AgentTrace.task_type,
            db.func.count(AgentTrace.trace_step_id).label('count')
        ).group_by(AgentTrace.task_type).all()
        
        return jsonify({
            "routing_stats": [
                {
                    "target_agent": stat[0],
                    "task_type": stat[1],
                    "total_routes": stat[2],
                    "successful_routes": stat[3],
                    "success_rate": round((stat[3] / stat[2]) * 100, 2) if stat[2] > 0 else 0,
                    "avg_execution_time": round(stat[4] or 0, 2)
                } for stat in routing_stats
            ],
            "task_distribution": [
                {
                    "task_type": stat[0],
                    "count": stat[1]
                } for stat in task_distribution
            ]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/admin/clear-chat-history', methods=['DELETE'])
def clear_chat_route():
    """Clear all chat history including multi-agent data"""
    return clear_chat_history()

@app.route('/api/admin/clear-session/<session_id>', methods=['DELETE'])
def clear_session_route(session_id):
    """Clear specific session including multi-agent data"""
    return clear_session_data(session_id)

@app.route('/api/tools/definitions', methods=['GET', 'POST'])
def handle_tool_definitions():
    """Handle tool definitions with multi-agent context"""
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
            cost_per_call_cents=data.get('cost_per_call_cents', 0),
            agent_type=data.get('agent_type', 'general')
        )
        db.session.add(tool_def)
        db.session.commit()
        return jsonify(tool_def.to_dict()), 201

@app.route('/api/agents/definitions', methods=['GET', 'POST'])
def handle_agent_definitions():
    """Handle agent definitions"""
    if request.method == 'GET':
        agents = AgentDefinition.query.all()
        return jsonify([agent.to_dict() for agent in agents])
    
    if request.method == 'POST':
        data = request.json
        agent_def = AgentDefinition(
            name=data['name'],
            description=data.get('description'),
            agent_type=data.get('agent_type', 'specialist'),
            llm_config=data.get('llm_config', {}),
            prompt_template=data.get('prompt_template', '')
        )
        db.session.add(agent_def)
        db.session.commit()
        return jsonify(agent_def.to_dict()), 201

# Enhanced endpoint for logging multi-agent traces
@app.route('/api/chat/log-multi-agent-trace', methods=['POST'])
def log_multi_agent_trace():
    """Log multi-agent trace with enhanced context"""
    import traceback

    try:
        data = request.json
        chat_manager = ChatHistoryManager(
            session_id=data.get('session_id'),
            user_id=data.get('user_id')
        )
        
        # Extract multi-agent context
        agent_used = data.get('agent_used', 'unknown')
        task_type = data.get('task_type', 'general')
        routing_info = data.get('routing_info', {})
        
        # Log the multi-agent trace
        result = chat_manager.add_multi_agent_trace(
            serialized_messages=data.get('messages'),
            trace_duration=data.get('trace_duration', 0),
            agent_used=agent_used,
            task_type=task_type,
            routing_info=routing_info
        )

        return jsonify({
            "status": "success",
            "message": result,
            "agent_used": agent_used,
            "task_type": task_type
        }), 201

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

# Legacy endpoint for backward compatibility
@app.route('/api/chat/log-trace', methods=['POST'])
def log_trace():
    """Legacy trace logging - redirects to multi-agent version"""
    import traceback

    try:
        data = request.json
        chat_manager = ChatHistoryManager(
            session_id=data.get('session_id'),
            user_id=data.get('user_id')
        )
        
        # Use the legacy method for backward compatibility
        result = chat_manager.add_trace_messages(
            serialized_messages=data.get('messages'),
            trace_duration=data.get('trace_duration', 0)
        )

        return jsonify({"status": "success", "message": result}), 201

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
    
# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check with multi-agent system status"""
    try:
        # Check database connectivity
        db.session.execute('SELECT 1')
        
        # Check agent definitions
        agent_count = AgentDefinition.query.count()
        tool_count = ToolDefinition.query.count()
        
        return jsonify({
            "status": "healthy", 
            "service": "multi-agent-analytics",
            "agents_defined": agent_count,
            "tools_defined": tool_count
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "service": "multi-agent-analytics", 
            "error": str(e)
        }), 500

if __name__ == '__main__':
    print("[Analytics Service] Connecting to database...")
    print("You may be prompted for credentials...")
    
    with app.app_context():
        db.create_all()
        print("[Analytics Service] Database initialized")
        initialize_tool_definitions()
        print("[Analytics Service] Multi-agent tool definitions initialized")
        initialize_agent_definitions()
        print("[Analytics Service] Multi-agent definitions initialized")
    
    print("Starting Multi-Agent Analytics Service on port 5002...")
    app.run(debug=False, port=5002, use_reloader=False)