# import urllib.parse
import uuid
from datetime import datetime
import json
import time
from dateutil.relativedelta import relativedelta
# from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_sqlserver import SQLServer_VectorStore
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.store.memory import InMemoryStore
from shared.connection_manager import sqlalchemy_connection_creator, connection_manager
from shared.utils import get_user_id
import requests  # For calling analytics service
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from shared.utils import _serialize_messages
from init_data import check_and_ingest_data
# Load Environment variables and initialize app
import os
load_dotenv(override=True)

app = Flask(__name__)
CORS(app)
global fixed_user_id
fixed_user_id = get_user_id()  # For simplicity, using a fixed user ID

# --- Azure OpenAI Configuration ---
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# Analytics service URL
ANALYTICS_SERVICE_URL = "http://127.0.0.1:5002"

if not all([AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_EMBEDDING_DEPLOYMENT]):
    print("⚠️  Warning: One or more Azure OpenAI environment variables are not set.")
    ai_client = None
    embeddings_client = None
else:
    ai_client = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version="2024-10-21",
        api_key=AZURE_OPENAI_KEY,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT 
    )
    embeddings_client = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        openai_api_version="2024-10-21",
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
    )

# Database configuration for Azure SQL (banking data)
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
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

connection_string = os.getenv('FABRIC_SQL_CONNECTION_URL_AGENTIC')

connection_url = f"mssql+pyodbc:///?odbc_connect={connection_string}"

vector_store = None
if embeddings_client:
    vector_store = SQLServer_VectorStore(
        connection_string=connection_url,
        table_name="DocsChunks_Embeddings",
        embedding_function=embeddings_client,
        embedding_length=1536,
        distance_strategy=DistanceStrategy.COSINE,
    )

def to_dict_helper(instance):
    d = {}
    for column in instance.__table__.columns:
        value = getattr(instance, column.name)
        if isinstance(value, datetime):
            d[column.name] = value.isoformat()
        else:
            d[column.name] = value
    return d

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from collections import defaultdict

def reconstruct_messages_from_history(history_data):
    """Converts DB history into LangChain message objects, sorted by trace_id and message order."""
    messages = []
    print("Reconstructing messages from history data:", history_data)
    
    if not history_data:
        return MemorySaver(), []
    
    # Group messages by trace_id
    traces = defaultdict(list)
    for msg_data in history_data:
        trace_id = msg_data.get('trace_id')
        if trace_id:
            traces[trace_id].append(msg_data)
    
    # Sort trace_ids chronologically
    sorted_trace_ids = sorted(traces.keys())
    
    # Process each trace in chronological order
    for trace_id in sorted_trace_ids:
        trace_messages = traces[trace_id]
        
        # Sort messages within each trace by message type priority
        message_priority = {
            'human': 1,
            'ai': 2
        }
        
        trace_messages.sort(key=lambda x: (
            message_priority.get(x.get('message_type'), 5),
            x.get('trace_end', ''),
        ))
        
        # Convert to LangChain message objects
        for msg_data in trace_messages:
            try:
                message_type = msg_data.get('message_type')
                content = msg_data.get('content', '')
                
                if message_type == 'human':
                    messages.append(HumanMessage(content=content))
                elif message_type == 'ai':
                    messages.append(AIMessage(content=content))
                
            except Exception as e:
                print(f"Error processing message in trace {trace_id}: {e}")
                continue
    
    print(f"Reconstructed {len(messages)} messages from {len(sorted_trace_ids)} traces")
    
    # Return both the memory saver and the historical messages
    return MemorySaver(), messages

# Banking Database Models
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.String(255), primary_key=True, default=lambda: f"user_{uuid.uuid4()}")
    name = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    accounts = db.relationship('Account', backref='user', lazy=True)

    def to_dict(self):
        return to_dict_helper(self)

class Account(db.Model):
    __tablename__ = 'accounts'
    id = db.Column(db.String(255), primary_key=True, default=lambda: f"acc_{uuid.uuid4()}")
    user_id = db.Column(db.String(255), db.ForeignKey('users.id'), nullable=False)
    account_number = db.Column(db.String(255), unique=True, nullable=False, default=lambda: str(uuid.uuid4().int)[:12])
    account_type = db.Column(db.String(50), nullable=False)
    balance = db.Column(db.Float, nullable=False)
    name = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return to_dict_helper(self)

class Transaction(db.Model):
    __tablename__ = 'transactions'
    id = db.Column(db.String(255), primary_key=True, default=lambda: f"txn_{uuid.uuid4()}")
    from_account_id = db.Column(db.String(255), db.ForeignKey('accounts.id'))
    to_account_id = db.Column(db.String(255), db.ForeignKey('accounts.id'))
    amount = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(50), nullable=False)
    description = db.Column(db.String(255))
    category = db.Column(db.String(255))
    status = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return to_dict_helper(self)

# Analytics Service Integration
def call_analytics_service(endpoint, method='POST', data=None):
    """Helper function to call analytics service"""
    try:
        url = f"{ANALYTICS_SERVICE_URL}/api/{endpoint}"
        if method == 'POST':
            response = requests.post(url, json=data, timeout=5)
        else:
            response = requests.get(url, timeout=5)
        return response.json() if response.status_code < 400 else None
    except Exception as e:
        print(f"Analytics service call failed: {e}")
        return None

import os
import json
import time
import uuid
from datetime import datetime
from typing import Annotated, TypedDict, List

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from dateutil.relativedelta import relativedelta

# Import existing banking infrastructure
from banking_app import (
    db, Account, Transaction, User, fixed_user_id, 
    vector_store, call_analytics_service, _serialize_messages,
    reconstruct_messages_from_history
)

# Multi-Agent State
class BankingAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    current_agent: str
    task_type: str
    user_id: str
    session_id: str
    final_result: str

# Specialized Banking Tools

# Account Management Tools
@tool
def get_user_accounts_tool(user_id: str = fixed_user_id) -> str:
    """Retrieves all accounts for a given user."""
    try:
        accounts = Account.query.filter_by(user_id=user_id).all()
        if not accounts:
            return "No accounts found for this user."
        return json.dumps([
            {"name": acc.name, "account_type": acc.account_type, "balance": acc.balance} 
            for acc in accounts
        ])
    except Exception as e:
        return f"Error retrieving accounts: {str(e)}"

@tool
def create_new_account_tool(user_id: str = fixed_user_id, account_type: str = 'checking', name: str = None, balance: float = 0.0) -> str:
    """Creates a new bank account for the user."""
    if not name:
        return json.dumps({"status": "error", "message": "An account name is required."})
    try:
        new_account = Account(user_id=user_id, account_type=account_type, balance=balance, name=name)
        db.session.add(new_account)
        db.session.commit()
        return json.dumps({
            "status": "success", 
            "message": f"Successfully created new {account_type} account '{name}' with balance ${balance:.2f}.",
            "account_id": new_account.id, 
            "account_name": new_account.name
        })
    except Exception as e:
        db.session.rollback()
        return f"Error creating account: {str(e)}"

# Transaction Tools
@tool
def transfer_money_tool(user_id: str = fixed_user_id, from_account_name: str = None, to_account_name: str = None, amount: float = 0.0, to_external_details: dict = None) -> str:
    """Transfers money between user's accounts or to an external account."""
    if not from_account_name or (not to_account_name and not to_external_details) or amount <= 0:
        return json.dumps({"status": "error", "message": "Missing required transfer details."})
    
    try:
        from_account = Account.query.filter_by(user_id=user_id, name=from_account_name).first()
        if not from_account:
            return json.dumps({"status": "error", "message": f"Account '{from_account_name}' not found."})
        if from_account.balance < amount:
            return json.dumps({"status": "error", "message": "Insufficient funds."})
        
        to_account = None
        if to_account_name:
            to_account = Account.query.filter_by(user_id=user_id, name=to_account_name).first()
            if not to_account:
                 return json.dumps({"status": "error", "message": f"Recipient account '{to_account_name}' not found."})
        
        new_transaction = Transaction(
            from_account_id=from_account.id, 
            to_account_id=to_account.id if to_account else None,
            amount=amount, 
            type='transfer', 
            description=f"Transfer to {to_account_name or to_external_details.get('name', 'External')}",
            category='Transfer', 
            status='completed'
        )
        
        from_account.balance -= amount
        if to_account:
            to_account.balance += amount
            
        db.session.add(new_transaction)
        db.session.commit()
        return json.dumps({"status": "success", "message": f"Successfully transferred ${amount:.2f}."})
    except Exception as e:
        db.session.rollback()
        return f"Error during transfer: {str(e)}"

@tool
def get_transactions_summary_tool(user_id: str = fixed_user_id, time_period: str = 'this month', account_name: str = None) -> str:
    """Provides a summary of the user's spending. Can be filtered by a time period and a specific account."""
    try:
        query = db.session.query(Transaction.category, db.func.sum(Transaction.amount).label('total_spent')).filter(
            Transaction.type == 'payment'
        )
        
        if account_name:
            account = Account.query.filter_by(user_id=user_id, name=account_name).first()
            if not account:
                return json.dumps({"status": "error", "message": f"Account '{account_name}' not found."})
            query = query.filter(Transaction.from_account_id == account.id)
        else:
            user_accounts = Account.query.filter_by(user_id=user_id).all()
            account_ids = [acc.id for acc in user_accounts]
            query = query.filter(Transaction.from_account_id.in_(account_ids))

        end_date = datetime.utcnow()
        if 'last 6 months' in time_period.lower():
            start_date = end_date - relativedelta(months=6)
        elif 'this year' in time_period.lower():
            start_date = end_date.replace(month=1, day=1, hour=0, minute=0, second=0)
        else:
            start_date = end_date.replace(day=1, hour=0, minute=0, second=0)
        
        query = query.filter(Transaction.created_at.between(start_date, end_date))
        results = query.group_by(Transaction.category).order_by(db.func.sum(Transaction.amount).desc()).all()
        total_spending = sum(r.total_spent for r in results)
        
        summary_details = {
            "total_spending": round(total_spending, 2),
            "period": time_period,
            "account_filter": account_name or "All Accounts",
            "top_categories": [{"category": r.category, "amount": round(r.total_spent, 2)} for r in results[:3]]
        }

        return json.dumps({"status": "success", "summary": summary_details})
    except Exception as e:
        return json.dumps({"status": "error", "message": f"An error occurred while generating the transaction summary."})

# Support Tools
@tool
def search_support_documents_tool(user_question: str) -> str:
    """Searches the knowledge base for answers to customer support questions using vector search."""
    if not vector_store:
        return "The vector store is not configured."
    try:
        results = vector_store.similarity_search_with_score(user_question, k=3)
        relevant_docs = [doc.page_content for doc, score in results if score < 0.5]
        
        if not relevant_docs:
            return "No relevant support documents found to answer this question."

        context = "\n\n---\n\n".join(relevant_docs)
        return context
    except Exception as e:
        return "An error occurred while searching for support documents."

# Create LLM
def create_banking_llm():
    return ai_client
# Specialized Banking Agents

def create_account_management_agent():
    """Agent specialized in account management operations."""
    llm = create_banking_llm()
    tools = [get_user_accounts_tool, create_new_account_tool]
    
    system_prompt = """You are an Account Management Agent for a banking system.
    
    Your responsibilities:
    1. Help customers view their accounts and account details
    2. Assist with creating new accounts (checking, savings, etc.)
    3. Provide account information and balances
    4. Handle account-related inquiries
    
    Always use the appropriate tools to get accurate, real-time account information.
    Be professional, secure, and helpful in all interactions.
    """
    
    return create_react_agent(
        llm, 
        tools, 
        prompt=system_prompt,
        checkpointer=MemorySaver()
    )

def create_transaction_agent():
    """Agent specialized in transaction operations."""
    llm = create_banking_llm()
    tools = [transfer_money_tool, get_transactions_summary_tool]
    
    system_prompt = """You are a Transaction Agent for a banking system.
    
    Your responsibilities:
    1. Process money transfers between accounts
    2. Provide transaction summaries and spending analysis
    3. Help customers understand their transaction history
    4. Assist with payment-related inquiries
    
    Always verify account details and ensure sufficient funds before processing transfers.
    Be careful with financial transactions and provide clear confirmations.
    """
    
    return create_react_agent(
        llm, 
        tools, 
        prompt=system_prompt,
        checkpointer=MemorySaver()
    )

def create_support_agent():
    """Agent specialized in customer support."""
    llm = create_banking_llm()
    tools = [search_support_documents_tool]
    
    system_prompt = """You are a Customer Support Agent for a banking system.
    
    Your responsibilities:
    1. Answer general banking questions using the knowledge base
    2. Provide information about banking policies and procedures
    3. Help customers with non-transactional inquiries
    4. Direct customers to appropriate specialists when needed
    
    Always search the support documents first to provide accurate information.
    Be helpful, empathetic, and professional in all customer interactions.
    """
    
    return create_react_agent(
        llm, 
        tools, 
        prompt=system_prompt,
        checkpointer=MemorySaver()
    )

def create_coordinator_agent():
    """Agent that routes customer requests to appropriate specialists."""
    llm = create_banking_llm()
    
    system_prompt = """You are a Banking Coordinator that routes customer requests to the right specialist.
    
    Route requests to:
    - account_agent: For account viewing, account creation, balance inquiries
    - transaction_agent: For money transfers, payment history, spending analysis
    - support_agent: For general questions, policies, troubleshooting
    
    Analyze the customer's request and respond with ONLY the agent name: 
    "account_agent", "transaction_agent", or "support_agent"
    """
    
    return create_react_agent(
        llm, 
        [], 
        prompt=system_prompt,
        checkpointer=MemorySaver()
    )

# Multi-Agent Node Functions

def coordinator_node(state: BankingAgentState):
    """Route customer requests to appropriate specialist agent."""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    # Enhanced routing logic
    message_lower = last_message.lower()
    
    # Account-related keywords
    account_keywords = ["account", "balance", "create account", "new account", "checking", "savings", "accounts"]
    
    # Transaction-related keywords  
    transaction_keywords = ["transfer", "send money", "payment", "transaction", "spending", "summary", "history"]
    
    # Support-related keywords
    support_keywords = ["help", "policy", "question", "how to", "what is", "explain", "support"]
    
    if any(keyword in message_lower for keyword in account_keywords):
        state["current_agent"] = "account_agent"
        state["task_type"] = "account_management"
    elif any(keyword in message_lower for keyword in transaction_keywords):
        state["current_agent"] = "transaction_agent"
        state["task_type"] = "transaction"
    else:
        state["current_agent"] = "support_agent"
        state["task_type"] = "support"
    
    return state

def account_agent_node(state: BankingAgentState):
    """Handle account management tasks."""
    account_agent = create_account_management_agent()
    
    thread_config = {"configurable": {"thread_id": f"account_{state['session_id']}"}}
    
    response = account_agent.invoke({"messages": state["messages"]}, config=thread_config)
    
    state["messages"] = response["messages"]
    state["final_result"] = response["messages"][-1].content
    
    return state

def transaction_agent_node(state: BankingAgentState):
    """Handle transaction-related tasks."""
    transaction_agent = create_transaction_agent()
    
    thread_config = {"configurable": {"thread_id": f"transaction_{state['session_id']}"}}
    
    response = transaction_agent.invoke({"messages": state["messages"]}, config=thread_config)
    
    state["messages"] = response["messages"]
    state["final_result"] = response["messages"][-1].content
    
    return state

def support_agent_node(state: BankingAgentState):
    """Handle customer support tasks."""
    support_agent = create_support_agent()
    
    thread_config = {"configurable": {"thread_id": f"support_{state['session_id']}"}}
    
    response = support_agent.invoke({"messages": state["messages"]}, config=thread_config)
    
    state["messages"] = response["messages"]
    state["final_result"] = response["messages"][-1].content
    
    return state

# Create Multi-Agent Banking System

def create_multi_agent_banking_system():
    """Create the multi-agent banking workflow."""
    
    workflow = StateGraph(BankingAgentState)
    
    # Add nodes
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("account_agent", account_agent_node)
    workflow.add_node("transaction_agent", transaction_agent_node)
    workflow.add_node("support_agent", support_agent_node)
    
    # Set entry point
    workflow.set_entry_point("coordinator")
    
    # Add conditional routing
    def route_to_specialist(state: BankingAgentState):
        return state["current_agent"]
    
    workflow.add_conditional_edges(
        "coordinator",
        route_to_specialist,
        {
            "account_agent": "account_agent",
            "transaction_agent": "transaction_agent",
            "support_agent": "support_agent"
        }
    )
    
    # All agents end the workflow
    workflow.add_edge("account_agent", END)
    workflow.add_edge("transaction_agent", END)
    workflow.add_edge("support_agent", END)
    
    return workflow.compile(checkpointer=MemorySaver())

# Multi-Agent Banking Chatbot Function

def multi_agent_banking_chatbot(messages, session_id, user_id=fixed_user_id):
    """Process banking requests using multi-agent system."""
    
    # Fetch chat history
    history_data = call_analytics_service(f"chat/history/{session_id}", method='GET')
    session_memory, historical_messages = reconstruct_messages_from_history(history_data)
    
    # Create multi-agent system
    banking_system = create_multi_agent_banking_system()
    
    # Extract current user message
    user_message = messages[-1].get("content", "")
    all_messages = historical_messages + [HumanMessage(content=user_message)]
    
    # Create initial state
    initial_state = {
        "messages": all_messages,
        "current_agent": "",
        "task_type": "",
        "user_id": user_id,
        "session_id": session_id,
        "final_result": ""
    }
    
    # Process with multi-agent system
    trace_start_time = time.time()
    result = banking_system.invoke(
        initial_state, 
        config={"configurable": {"thread_id": session_id}}
    )
    end_time = time.time()
    trace_duration = int((end_time - trace_start_time) * 1000)
    
    # Log analytics
    final_messages = result["messages"][len(historical_messages):]
    analytics_data = {
            "session_id": session_id,
            "user_id": user_id,
            "messages": _serialize_messages(final_messages),
            "trace_duration": trace_duration,
            "agent_used": result["current_agent"],
            "task_type": result["task_type"],
            "routing_info": {
                "coordinator": "coordinator_agent",
                "target_agent": result["current_agent"],
                "reason": f"Routed to {result['current_agent']} for {result['task_type']} task"
            }
        }
    
    call_analytics_service("chat/log-multi-agent-trace", data=analytics_data)
    
    return {
        "response": result["final_result"],
        "session_id": session_id,
        "agent_used": result["current_agent"],
        "task_type": result["task_type"],
        "tools_used": []
    }


# Banking API Routes
@app.route('/api/accounts', methods=['GET', 'POST'])
def handle_accounts():
    user_id = fixed_user_id
    if request.method == 'GET':
        accounts = Account.query.filter_by(user_id=user_id).all()
        return jsonify([acc.to_dict() for acc in accounts])
    if request.method == 'POST':
        data = request.json
        account_str = create_new_account_tool(user_id=user_id, account_type=data.get('account_type'), name=data.get('name'), balance=data.get('balance', 0))
        return jsonify(json.loads(account_str)), 201
    
@app.route('/api/transactions', methods=['GET', 'POST'])
def handle_transactions():
    user_id = fixed_user_id
    if request.method == 'GET':
        accounts = Account.query.filter_by(user_id=user_id).all()
        account_ids = [acc.id for acc in accounts]
        transactions = Transaction.query.filter((Transaction.from_account_id.in_(account_ids)) | (Transaction.to_account_id.in_(account_ids))).order_by(Transaction.created_at.desc()).all()
        return jsonify([t.to_dict() for t in transactions])
    if request.method == 'POST':
        data = request.json
        result_str = transfer_money_tool(
            user_id=user_id, from_account_name=data.get('from_account_name'), to_account_name=data.get('to_account_name'),
            amount=data.get('amount'), to_external_details=data.get('to_external_details')
        )
        result = json.loads(result_str)
        status_code = 201 if result.get("status") == "success" else 400
        return jsonify(result), status_code

# Replace the existing chatbot route with:
@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    if not ai_client:
        return jsonify({"error": "Azure OpenAI client is not configured."}), 503

    data = request.json
    messages = data.get("messages", [])
    session_id = data.get("session_id")
    user_id = fixed_user_id
    
    try:
        # Use multi-agent banking system
        result = multi_agent_banking_chatbot(messages, session_id, user_id)
        
        print(f"[Multi-Agent] Used: {result['agent_used']} for {result['task_type']}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Multi-agent banking error: {e}")
        return jsonify({"error": str(e)}), 500
    
def initialize_banking_app():
    """Initialize banking app when called from combined launcher."""
    with app.app_context():
        db.create_all()
        print("[Banking Service] Database tables initialized")