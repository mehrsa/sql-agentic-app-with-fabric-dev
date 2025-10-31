import uuid
from datetime import datetime
import json
from flask import jsonify
from shared.utils import _to_json_primitive
from shared.utils import get_user_id

# Global variables that will be set by the main app
db = None
ChatHistory = None
ChatSession = None
ToolUsage = None
ToolDefinition = None
ChatHistoryManager = None

def init_chat_db(database):
    """Initialize the database reference and create models"""
    global db, ChatHistory, ChatSession, ToolUsage, ToolDefinition, ChatHistoryManager, AgentDefinition, AgentTrace

    db = database

    # Helper function to convert model instances to dictionaries
    def to_dict_helper(instance):
        d = {}
        for column in instance.__table__.columns:
            value = getattr(instance, column.name)
            if isinstance(value, datetime):
                d[column.name] = value.isoformat()
            else:
                d[column.name] = value
        return d
    
    class AgentDefinition(db.Model):
        __tablename__ = 'agent_definitions'
        agent_id = db.Column(db.String(255), primary_key=True, default=lambda: f"agent_{uuid.uuid4()}")
        name = db.Column(db.String(255), unique=True, nullable=False)
        description = db.Column(db.Text)
        llm_config = db.Column(db.JSON, nullable=False)
        prompt_template = db.Column(db.Text, nullable=False)
        agent_type = db.Column(db.String(100), default='specialist')  # coordinator, specialist, support
        created_at = db.Column(db.DateTime, default=datetime.now())

        def to_dict(self):
            return to_dict_helper(self)

    class AgentTrace(db.Model):
        """Track agent routing and execution in multi-agent scenarios"""
        __tablename__ = 'agent_traces'
        trace_step_id = db.Column(db.String(255), primary_key=True, default=lambda: f"step_{uuid.uuid4()}")
        session_id = db.Column(db.String(255), db.ForeignKey('chat_sessions.session_id'), nullable=False)
        trace_id = db.Column(db.String(255), nullable=False)
        user_id = db.Column(db.String(255), nullable=False)
        
        # Multi-agent specific fields
        coordinator_agent = db.Column(db.String(255))  # Which coordinator routed this
        target_agent = db.Column(db.String(255))       # Which agent was selected
        routing_reason = db.Column(db.Text)             # Why this agent was chosen
        task_type = db.Column(db.String(100))          # Type of task (account, transaction, support)
        
        # Execution tracking
        step_order = db.Column(db.Integer, default=1)
        execution_start = db.Column(db.DateTime, default=datetime.now())
        execution_end = db.Column(db.DateTime)
        execution_duration_ms = db.Column(db.Integer)
        
        # Result tracking
        success = db.Column(db.Boolean, default=True)
        error_message = db.Column(db.Text)
        
        def to_dict(self):
            return to_dict_helper(self)

    class ChatSession(db.Model):
        __tablename__ = 'chat_sessions'
        session_id = db.Column(db.String(255), primary_key=True, default=lambda: f"session_{uuid.uuid4()}")
        user_id = db.Column(db.String(255), nullable=False)
        title = db.Column(db.String(500))
        
        # Multi-agent session tracking
        total_agents_used = db.Column(db.Integer, default=0)
        primary_agent_type = db.Column(db.String(100))  # Most used agent type
        
        created_at = db.Column(db.DateTime, default=datetime.now())
        updated_at = db.Column(db.DateTime, default=datetime.now(), onupdate=datetime.now())

        def to_dict(self):
            return to_dict_helper(self)
        
    class ToolDefinition(db.Model):
        __tablename__ = 'tool_definitions'
        tool_id = db.Column(db.String(255), primary_key=True, default=lambda: f"tooldef_{uuid.uuid4()}")
        name = db.Column(db.String(255), unique=True, nullable=False)
        description = db.Column(db.Text)
        input_schema = db.Column(db.JSON, nullable=False)
        version = db.Column(db.String(50), default='1.0.0')
        is_active = db.Column(db.Boolean, default=True)
        cost_per_call_cents = db.Column(db.Integer, default=0)
        
        # Multi-agent tool association
        agent_type = db.Column(db.String(100))  # Which agent type uses this tool
        
        created_at = db.Column(db.DateTime, default=datetime.now())
        updated_at = db.Column(db.DateTime, default=datetime.now(), onupdate=datetime.now())

        def to_dict(self):
            return to_dict_helper(self)
        
    class ToolUsage(db.Model):
        __tablename__ = 'tool_usage'
        tool_call_id = db.Column(db.String(255), primary_key=True, default=lambda: f"tool_{uuid.uuid4()}")
        session_id = db.Column(db.String(255), nullable=False)
        trace_id = db.Column(db.String(255), db.ForeignKey('chat_history.trace_id'))
        tool_id = db.Column(db.String(255), db.ForeignKey('tool_definitions.tool_id'), nullable=False)
        tool_name = db.Column(db.String(255), nullable=False)
        tool_input = db.Column(db.JSON, nullable=False)
        tool_output = db.Column(db.JSON)
        tool_message = db.Column(db.Text)
        status = db.Column(db.String(50))
        
        # Multi-agent context
        executing_agent = db.Column(db.String(255))  # Which agent executed this tool
        agent_type = db.Column(db.String(100))       # Type of agent that used this tool
        
        # Additional tracking fields
        tokens_used = db.Column(db.Integer)
        execution_time_ms = db.Column(db.Integer)

        def to_dict(self):
            return to_dict_helper(self)

    class ChatHistory(db.Model):
        __tablename__ = 'chat_history'
        message_id = db.Column(db.String(255), primary_key=True, default=lambda: f"msg_{uuid.uuid4()}")
        session_id = db.Column(db.String(255), db.ForeignKey('chat_sessions.session_id'))
        trace_id = db.Column(db.String(255), nullable=False)
        user_id = db.Column(db.String(255), nullable=False)
        
        # Multi-agent tracking
        agent_id = db.Column(db.String(255), nullable=True)
        agent_name = db.Column(db.String(255))       # Name of the agent
        agent_type = db.Column(db.String(100))       # Type: coordinator, account_agent, transaction_agent, support_agent
        routing_step = db.Column(db.Integer)         # Step in multi-agent flow
        
        message_type = db.Column(db.String(50), nullable=False)  # 'human', 'ai', 'system', 'tool_call', 'tool_result', 'routing'
        content = db.Column(db.Text)

        # LLM metadata
        model_name = db.Column(db.String(255))
        content_filter_results = db.Column(db.JSON)
        total_tokens = db.Column(db.Integer)
        completion_tokens = db.Column(db.Integer)
        prompt_tokens = db.Column(db.Integer)

        # Tool information
        tool_id = db.Column(db.String(255))
        tool_name = db.Column(db.String(255))
        tool_input = db.Column(db.JSON)
        tool_output = db.Column(db.JSON) 
        tool_call_id = db.Column(db.String(255))
    
        finish_reason = db.Column(db.String(255))
        response_time_ms = db.Column(db.Integer)
        trace_end = db.Column(db.DateTime, default=datetime.now())

        def to_dict(self):
            return to_dict_helper(self)

    # --- Enhanced Chat History Management Class ---
    class ChatHistoryManager:
        def __init__(self, session_id: str, user_id: str = 'user_1'):
            self.session_id = session_id
            self.user_id = user_id
            self._ensure_session_exists()

        def _ensure_session_exists(self):
            """Ensure the chat session exists in the database"""
            session = ChatSession.query.filter_by(session_id=self.session_id).first()
            if not session:
                session = ChatSession(
                    session_id=self.session_id,
                    title="New Multi-Agent Session",
                    user_id=self.user_id,
                )
                print("-----------------> New multi-agent chat session created: ", session.session_id)
                db.session.add(session)
                db.session.commit()

        def add_multi_agent_trace(self, serialized_messages: str, trace_duration: int, 
                                agent_used: str = None, task_type: str = None, routing_info: dict = None):
            """Add all messages in a multi-agent trace to the chat history"""
            trace_id = str(uuid.uuid4())
            message_list = _to_json_primitive(serialized_messages)
            
            print(f"New multi-agent trace_id generated: {trace_id}")
            print(f"Agent used: {agent_used}, Task type: {task_type}")
            
            # Log agent routing information
            if routing_info:
                self._log_agent_routing(trace_id, routing_info, task_type)
            
            routing_step = 1
            for msg in message_list:
                if msg['type'] == 'human':
                    print("Adding human message to chat history")
                    self.add_human_message(msg, trace_id, routing_step)
                    routing_step += 1
                elif msg['type'] == 'ai':
                    print(f"Adding AI message to chat history from agent: {agent_used}")
                    if msg.get("response_metadata", {}).get("finish_reason") != "tool_calls":
                        self.add_ai_message(msg, trace_id, trace_duration, agent_used, task_type, routing_step)
                    else:
                        tool_call_dict = self.add_tool_call_message(msg, trace_id, agent_used, routing_step)
                    routing_step += 1
                elif msg['type'] == "tool":
                    print(f"Adding tool message to chat history for agent: {agent_used}")
                    tool_result_dict = self.add_tool_result_message(msg, trace_id, agent_used, routing_step)
                    if 'tool_call_dict' in locals():
                        tool_call_dict.update(tool_result_dict)
                        self.log_tool_usage(tool_call_dict, trace_id, agent_used, task_type)
                    routing_step += 1
            
            self.update_session_stats(agent_used, task_type)
            return "All multi-agent trace messages added..."

        def _log_agent_routing(self, trace_id: str, routing_info: dict, task_type: str):
            """Log agent routing decisions"""
            agent_trace = AgentTrace(
                session_id=self.session_id,
                trace_id=trace_id,
                user_id=self.user_id,
                coordinator_agent=routing_info.get('coordinator', 'system'),
                target_agent=routing_info.get('target_agent'),
                routing_reason=routing_info.get('reason', 'Automated routing'),
                task_type=task_type,
                execution_start=datetime.now()
            )
            db.session.add(agent_trace)
            db.session.commit()

        def add_human_message(self, message: dict, trace_id: str, routing_step: int):
            """Add the human message to chat history"""
            entry_message = ChatHistory(
                session_id=self.session_id,
                user_id=self.user_id,
                trace_id=trace_id,
                message_id=str(uuid.uuid4()),
                message_type="human",
                content=message['content'],
                routing_step=routing_step,
                agent_type='user'
            )
            db.session.add(entry_message)
            db.session.commit()
            print("Human message added to chat history:", message.get("id", "unknown"))
            return entry_message

        def add_ai_message(self, message: dict, trace_id: str, trace_duration: int, 
                          agent_used: str = None, task_type: str = None, routing_step: int = 1):
            """Add the AI agent message to chat history with multi-agent context"""
            agent_id = None
            if "name" in message:
                agent_id = db.session.query(AgentDefinition.agent_id).filter_by(name=message["name"]).scalar()
            
            entry_message = ChatHistory(
                session_id=self.session_id,
                user_id=self.user_id,
                agent_id=agent_id,
                agent_name=agent_used,
                agent_type=task_type,
                message_id=message.get("id", str(uuid.uuid4())),
                trace_id=trace_id,
                message_type="ai",
                content=message["content"],
                routing_step=routing_step,
                total_tokens=message.get("response_metadata", {}).get("token_usage", {}).get('total_tokens'),
                completion_tokens=message.get("response_metadata", {}).get("token_usage", {}).get('completion_tokens'),
                prompt_tokens=message.get("response_metadata", {}).get("token_usage", {}).get('prompt_tokens'),
                model_name=message.get("response_metadata", {}).get('model_name'),
                content_filter_results=message.get("response_metadata", {}).get("prompt_filter_results", [{}])[0].get("content_filter_results"),
                finish_reason=message.get("response_metadata", {}).get("finish_reason"),
                response_time_ms=trace_duration,
            )
            db.session.add(entry_message)
            db.session.commit()
            print(f"AI message added to chat history from {agent_used}: {message.get('id', 'unknown')}")
            return entry_message

        def add_tool_call_message(self, message: dict, trace_id: str, agent_used: str, routing_step: int):
            """Log a tool call with multi-agent context"""
            agent_id = None
            if "name" in message:
                agent_id = db.session.query(AgentDefinition.agent_id).filter_by(name=message["name"]).scalar()
            
            tool_name = message.get("additional_kwargs", {}).get('tool_calls', [{}])[0].get('function', {}).get("name")
            tool_id = db.session.query(ToolDefinition.tool_id).filter_by(name=tool_name).scalar()

            entry_message = ChatHistory(
                session_id=self.session_id,
                user_id=self.user_id,
                agent_id=agent_id,
                agent_name=agent_used,
                trace_id=trace_id,
                message_type='tool_call',
                routing_step=routing_step,
                tool_id=tool_id,
                tool_call_id=message.get("additional_kwargs", {}).get('tool_calls', [{}])[0].get('id'),
                tool_name=tool_name,
                total_tokens=message.get("response_metadata", {}).get("token_usage", {}).get('total_tokens'),
                completion_tokens=message.get("response_metadata", {}).get("token_usage", {}).get('completion_tokens'),
                prompt_tokens=message.get("response_metadata", {}).get("token_usage", {}).get('prompt_tokens'),
                tool_input=message.get("additional_kwargs", {}).get('tool_calls', [{}])[0].get('function', {}).get("arguments"),
                model_name=message.get("response_metadata", {}).get('model_name'),
                content_filter_results=message.get("response_metadata", {}).get("prompt_filter_results", [{}])[0].get("content_filter_results"),
                finish_reason=message.get("response_metadata", {}).get("finish_reason"),
            )
            db.session.add(entry_message)
            db.session.commit()
            print(f"Tool call message added to chat history for {agent_used}: {message.get('id', 'unknown')}")
            
            return {
                "tool_call_id": message.get("additional_kwargs", {}).get('tool_calls', [{}])[0].get('id'),
                "tool_id": tool_id, 
                "tool_name": tool_name,
                "tool_input": message.get("additional_kwargs", {}).get('tool_calls', [{}])[0].get('function', {}).get("arguments"),
                "total_tokens": message.get("response_metadata", {}).get("token_usage", {}).get('total_tokens'),
                "executing_agent": agent_used
            }

        def add_tool_result_message(self, message: dict, trace_id: str, agent_used: str, routing_step: int):
            """Log a tool result with multi-agent context"""
            tool_name = message["name"]
            tool_id = db.session.query(ToolDefinition.tool_id).filter_by(name=tool_name).scalar()
            
            entry_message = ChatHistory(
                session_id=self.session_id,
                user_id=self.user_id,
                message_id=message.get("id", str(uuid.uuid4())),
                tool_id=tool_id,
                tool_call_id=message["tool_call_id"],
                trace_id=trace_id,
                routing_step=routing_step,
                agent_name=agent_used,
                tool_name=message["name"],
                message_type='tool_result',
                content="",
                tool_output=message["content"],
            )
            db.session.add(entry_message)
            db.session.commit()
            print(f"Tool result message added for {agent_used}: {message.get('id', 'unknown')}")
            
            return {
                "tool_output": message["content"], 
                "status": message.get("status", "completed")
            }
        
        def update_session_stats(self, agent_used: str, task_type: str):
            """Update session statistics for multi-agent usage"""
            session = ChatSession.query.filter_by(session_id=self.session_id).first()
            if session:
                session.updated_at = datetime.now()
                session.total_agents_used = session.total_agents_used + 1 if session.total_agents_used else 1
                
                # Update primary agent type if this is the most used
                if not session.primary_agent_type:
                    session.primary_agent_type = task_type
                
                db.session.commit()
                print(f"Session stats updated - Agent: {agent_used}, Total agents used: {session.total_agents_used}")
     
        def log_tool_usage(self, tool_info: dict, trace_id: str, agent_used: str, task_type: str):
            """Log detailed tool usage metrics with multi-agent context"""
            existing = ToolUsage.query.filter_by(tool_call_id=tool_info.get("tool_call_id")).first()
            
            tool_msg = ''
            if isinstance(tool_info.get("tool_output"), dict):
                tool_msg = tool_info.get("tool_output").get('message', '')
            else:
                tool_msg = str(tool_info.get("tool_output"))
            
            tool_call_status = "Error" if "error" in str(tool_info.get("tool_output")).lower() else "Success"

            if existing:
                existing.tool_output = tool_info.get("tool_output")
                existing.trace_id = trace_id
                existing.session_id = self.session_id
                existing.tool_id = tool_info.get("tool_id") 
                existing.tool_name = tool_info.get("tool_name") 
                existing.tool_input = tool_info.get("tool_input") 
                existing.tool_output = tool_info.get("tool_output") 
                existing.tool_message = tool_msg
                existing.status = tool_call_status
                existing.tokens_used = tool_info.get("total_tokens")
                existing.executing_agent = agent_used
                existing.agent_type = task_type
                db.session.commit()
            else:
                tool_usage = ToolUsage(
                    session_id=self.session_id,
                    trace_id=trace_id,
                    tool_call_id=tool_info.get("tool_call_id"),
                    tool_id=tool_info.get("tool_id"),
                    tool_name=tool_info.get("tool_name"),
                    tool_input=tool_info.get("tool_input"),
                    tool_output=tool_info.get("tool_output"),
                    tool_message=tool_msg,
                    status=tool_call_status,
                    tokens_used=tool_info.get("total_tokens"),
                    executing_agent=agent_used,
                    agent_type=task_type
                )
                db.session.add(tool_usage)
                db.session.commit()

        # Legacy method for backward compatibility
        def add_trace_messages(self, serialized_messages: str, trace_duration: int):
            """Legacy method - redirects to multi-agent version"""
            return self.add_multi_agent_trace(serialized_messages, trace_duration)

        def get_conversation_history(self):
            """Get conversation history with multi-agent context"""
            messages = ChatHistory.query.filter_by(session_id=self.session_id).order_by(ChatHistory.trace_end).all()
            return [msg.to_dict() for msg in messages]

        def get_agent_usage_stats(self):
            """Get statistics about agent usage in this session"""
            agent_stats = db.session.query(
                ChatHistory.agent_type,
                db.func.count(ChatHistory.message_id).label('message_count'),
                db.func.sum(ChatHistory.total_tokens).label('total_tokens')
            ).filter_by(
                session_id=self.session_id,
                message_type='ai'
            ).group_by(ChatHistory.agent_type).all()
            
            return [{"agent_type": stat[0], "message_count": stat[1], "total_tokens": stat[2] or 0} 
                   for stat in agent_stats]
        
    # Make classes available globally in this module
    globals()['ChatHistory'] = ChatHistory
    globals()['ChatSession'] = ChatSession
    globals()['ToolUsage'] = ToolUsage
    globals()['ToolDefinition'] = ToolDefinition
    globals()['ChatHistoryManager'] = ChatHistoryManager


def handle_chat_sessions(request):
    if request.method == 'GET':
        sessions = ChatSession.query.order_by(ChatSession.updated_at.desc()).all()
        return jsonify([session.to_dict() for session in sessions])
    
    if request.method == 'POST':
        data = request.json
        user_id = data.get('user_id', get_user_id())
        session = ChatSession(
            user_id=user_id,
            title=data.get('title', 'New Multi-Agent Session')
        )
        db.session.add(session)
        db.session.commit()
        return jsonify(session.to_dict()), 201

def clear_chat_history():
    try:
        db.session.query(ChatHistory).delete()
        db.session.query(ToolUsage).delete()
        db.session.query(AgentTrace).delete()
        db.session.commit()
        return jsonify({"message": "All chat history and agent traces cleared"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

def clear_session_data(session_id):
    try:
        db.session.query(ChatHistory).filter_by(session_id=session_id).delete()
        db.session.query(ToolUsage).filter_by(session_id=session_id).delete()
        db.session.query(AgentTrace).filter_by(session_id=session_id).delete()
        db.session.query(ChatSession).filter_by(session_id=session_id).delete()
        db.session.commit()
        return jsonify({"message": f"Session {session_id} cleared"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

def initialize_tool_definitions():
    """Initialize tool definitions for multi-agent banking system"""
    tools = [
        {
            "name": "get_user_accounts_tool",
            "description": "Retrieves all accounts for a given user",
            "input_schema": {"user_id": "string"},
            "agent_type": "account_agent"
        },
        {
            "name": "create_new_account_tool", 
            "description": "Creates a new bank account for the user",
            "input_schema": {"user_id": "string", "account_type": "string", "name": "string", "balance": "number"},
            "agent_type": "account_agent"
        },
        {
            "name": "transfer_money_tool",
            "description": "Transfers money between user accounts",
            "input_schema": {"user_id": "string", "from_account_name": "string", "to_account_name": "string", "amount": "number"},
            "agent_type": "transaction_agent"
        },
        {
            "name": "get_transactions_summary_tool",
            "description": "Provides a summary of user spending",
            "input_schema": {"user_id": "string", "time_period": "string", "account_name": "string"},
            "agent_type": "transaction_agent"
        },
        {
            "name": "search_support_documents_tool",
            "description": "Searches the knowledge base for customer support",
            "input_schema": {"user_question": "string"},
            "agent_type": "support_agent"
        }
    ]
    
    for tool_data in tools:
        existing = ToolDefinition.query.filter_by(name=tool_data["name"]).first()
        if not existing:
            tool = ToolDefinition(
                name=tool_data["name"],
                description=tool_data["description"],
                input_schema=tool_data["input_schema"],
                agent_type=tool_data["agent_type"]
            )
            db.session.add(tool)
    
    db.session.commit()
    print("Multi-agent tool definitions initialized")

def initialize_agent_definitions():
    """Initialize agent definitions for multi-agent banking system"""
    agents = [
        {
            "name": "coordinator_agent",
            "description": "Routes customer requests to appropriate specialist agents",
            "agent_type": "coordinator",
            "llm_config": {"model": "gpt-4", "temperature": 0.1},
            "prompt_template": "You are a Banking Coordinator that routes customer requests to the right specialist."
        },
        {
            "name": "account_agent", 
            "description": "Specialized in account management operations",
            "agent_type": "specialist",
            "llm_config": {"model": "gpt-4", "temperature": 0.1},
            "prompt_template": "You are an Account Management Agent for a banking system."
        },
        {
            "name": "transaction_agent",
            "description": "Specialized in transaction operations",
            "agent_type": "specialist", 
            "llm_config": {"model": "gpt-4", "temperature": 0.1},
            "prompt_template": "You are a Transaction Agent for a banking system."
        },
        {
            "name": "support_agent",
            "description": "Specialized in customer support",
            "agent_type": "support",
            "llm_config": {"model": "gpt-4", "temperature": 0.1},
            "prompt_template": "You are a Customer Support Agent for a banking system."
        }
    ]
    
    for agent_data in agents:
        existing = AgentDefinition.query.filter_by(name=agent_data["name"]).first()
        if not existing:
            agent = AgentDefinition(
                name=agent_data["name"],
                description=agent_data["description"],
                agent_type=agent_data["agent_type"],
                llm_config=agent_data["llm_config"],
                prompt_template=agent_data["prompt_template"]
            )
            db.session.add(agent)
    
    db.session.commit()
    print("Multi-agent definitions initialized")