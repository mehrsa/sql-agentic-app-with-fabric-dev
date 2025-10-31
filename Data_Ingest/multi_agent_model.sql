-- ===================================================================
-- Multi-Agent Banking System - Database Initialization Script
-- This script creates all required tables for the enhanced multi-agent
-- banking system with comprehensive analytics and tracking capabilities
-- ===================================================================

-- Drop existing tables if they exist (in correct order to respect foreign keys)
IF OBJECT_ID('dbo.agent_traces', 'U') IS NOT NULL DROP TABLE dbo.agent_traces;
IF OBJECT_ID('dbo.tool_usage', 'U') IS NOT NULL DROP TABLE dbo.tool_usage;
IF OBJECT_ID('dbo.chat_history', 'U') IS NOT NULL DROP TABLE dbo.chat_history;
IF OBJECT_ID('dbo.chat_sessions', 'U') IS NOT NULL DROP TABLE dbo.chat_sessions;
IF OBJECT_ID('dbo.tool_definitions', 'U') IS NOT NULL DROP TABLE dbo.tool_definitions;
IF OBJECT_ID('dbo.agent_definitions', 'U') IS NOT NULL DROP TABLE dbo.agent_definitions;

GO


-- Users table
CREATE TABLE dbo.users (
    id NVARCHAR(255) PRIMARY KEY DEFAULT ('user_' + CAST(NEWID() AS NVARCHAR(36))),
    name NVARCHAR(255) NOT NULL,
    email NVARCHAR(255) UNIQUE NOT NULL,
    created_at DATETIME2(7) DEFAULT GETUTCDATE()
);

-- Accounts table
CREATE TABLE dbo.accounts (
    id NVARCHAR(255) PRIMARY KEY DEFAULT ('acc_' + CAST(NEWID() AS NVARCHAR(36))),
    user_id NVARCHAR(255) NOT NULL,
    account_number NVARCHAR(255) UNIQUE NOT NULL DEFAULT (LEFT(CAST(ABS(CHECKSUM(NEWID())) AS NVARCHAR(12)), 12)),
    account_type NVARCHAR(50) NOT NULL CHECK (account_type IN ('checking', 'savings', 'credit', 'investment', 'loan')),
    balance DECIMAL(18,2) NOT NULL DEFAULT 0.00,
    name NVARCHAR(255) NOT NULL,
    created_at DATETIME2(7) DEFAULT GETUTCDATE(),
    FOREIGN KEY (user_id) REFERENCES dbo.users(id)
);

-- Transactions table
CREATE TABLE dbo.transactions (
    id NVARCHAR(255) PRIMARY KEY DEFAULT ('txn_' + CAST(NEWID() AS NVARCHAR(36))),
    from_account_id NVARCHAR(255),
    to_account_id NVARCHAR(255),
    amount DECIMAL(18,2) NOT NULL,
    type NVARCHAR(50) NOT NULL CHECK (type IN ('payment', 'deposit', 'transfer', 'withdrawal', 'fee')),
    description NVARCHAR(255),
    category NVARCHAR(255),
    status NVARCHAR(50) NOT NULL DEFAULT 'completed' CHECK (status IN ('pending', 'completed', 'failed', 'cancelled')),
    created_at DATETIME2(7) DEFAULT GETUTCDATE(),
    FOREIGN KEY (from_account_id) REFERENCES dbo.accounts(id),
    FOREIGN KEY (to_account_id) REFERENCES dbo.accounts(id)
);

GO

-- ===================================================================
-- MULTI-AGENT SYSTEM TABLES
-- ===================================================================

-- Agent Definitions table
CREATE TABLE dbo.agent_definitions (
    agent_id NVARCHAR(255) PRIMARY KEY DEFAULT ('agent_' + CAST(NEWID() AS NVARCHAR(36))),
    name NVARCHAR(255) UNIQUE NOT NULL,
    description NVARCHAR(MAX),
    llm_config NVARCHAR(MAX) NOT NULL, -- JSON string
    prompt_template NVARCHAR(MAX) NOT NULL,
    agent_type NVARCHAR(100) DEFAULT 'specialist' CHECK (agent_type IN ('coordinator', 'specialist', 'support', 'system')),
    created_at DATETIME2(7) DEFAULT GETUTCDATE(),
    updated_at DATETIME2(7) DEFAULT GETUTCDATE()
);

-- Tool Definitions table
CREATE TABLE dbo.tool_definitions (
    tool_id NVARCHAR(255) PRIMARY KEY DEFAULT ('tooldef_' + CAST(NEWID() AS NVARCHAR(36))),
    name NVARCHAR(255) UNIQUE NOT NULL,
    description NVARCHAR(MAX),
    input_schema NVARCHAR(MAX) NOT NULL, -- JSON string
    version NVARCHAR(50) DEFAULT '1.0.0',
    is_active BIT DEFAULT 1,
    cost_per_call_cents INT DEFAULT 0,
    agent_type NVARCHAR(100), -- Which agent type primarily uses this tool
    created_at DATETIME2(7) DEFAULT GETUTCDATE(),
    updated_at DATETIME2(7) DEFAULT GETUTCDATE()
);

-- Chat Sessions table
CREATE TABLE dbo.chat_sessions (
    session_id NVARCHAR(255) PRIMARY KEY DEFAULT ('session_' + CAST(NEWID() AS NVARCHAR(36))),
    user_id NVARCHAR(255) NOT NULL,
    title NVARCHAR(500),
    total_agents_used INT DEFAULT 0,
    primary_agent_type NVARCHAR(100), -- Most frequently used agent type in this session
    created_at DATETIME2(7) DEFAULT GETUTCDATE(),
    updated_at DATETIME2(7) DEFAULT GETUTCDATE(),
    FOREIGN KEY (user_id) REFERENCES dbo.users(id)
);

-- Chat History table (enhanced for multi-agent tracking)
CREATE TABLE dbo.chat_history (
    message_id NVARCHAR(255) PRIMARY KEY DEFAULT ('msg_' + CAST(NEWID() AS NVARCHAR(36))),
    session_id NVARCHAR(255),
    trace_id NVARCHAR(255) NOT NULL,
    user_id NVARCHAR(255) NOT NULL,
    
    -- Multi-agent tracking
    agent_id NVARCHAR(255),
    agent_name NVARCHAR(255),
    agent_type NVARCHAR(100), -- coordinator, account_agent, transaction_agent, support_agent
    routing_step INT, -- Step number in multi-agent flow
    
    message_type NVARCHAR(50) NOT NULL CHECK (message_type IN ('human', 'ai', 'system', 'tool_call', 'tool_result', 'routing')),
    content NVARCHAR(MAX),

    -- LLM metadata
    model_name NVARCHAR(255),
    content_filter_results NVARCHAR(MAX), -- JSON string
    total_tokens INT,
    completion_tokens INT,
    prompt_tokens INT,

    -- Tool information
    tool_id NVARCHAR(255),
    tool_name NVARCHAR(255),
    tool_input NVARCHAR(MAX), -- JSON string
    tool_output NVARCHAR(MAX), -- JSON string
    tool_call_id NVARCHAR(255),
    
    finish_reason NVARCHAR(255),
    response_time_ms INT,
    trace_end DATETIME2(7) DEFAULT GETUTCDATE(),
    
    FOREIGN KEY (session_id) REFERENCES dbo.chat_sessions(session_id),
    FOREIGN KEY (agent_id) REFERENCES dbo.agent_definitions(agent_id),
    FOREIGN KEY (tool_id) REFERENCES dbo.tool_definitions(tool_id),
    FOREIGN KEY (user_id) REFERENCES dbo.users(id)
);

-- Agent Traces table (for tracking multi-agent routing and execution)
CREATE TABLE dbo.agent_traces (
    trace_step_id NVARCHAR(255) PRIMARY KEY DEFAULT ('step_' + CAST(NEWID() AS NVARCHAR(36))),
    session_id NVARCHAR(255) NOT NULL,
    trace_id NVARCHAR(255) NOT NULL,
    user_id NVARCHAR(255) NOT NULL,
    
    -- Multi-agent routing information
    coordinator_agent NVARCHAR(255), -- Which coordinator made the routing decision
    target_agent NVARCHAR(255), -- Which agent was selected for execution
    routing_reason NVARCHAR(MAX), -- Why this agent was chosen
    task_type NVARCHAR(100), -- account, transaction, support, etc.
    
    -- Execution tracking
    step_order INT DEFAULT 1,
    execution_start DATETIME2(7) DEFAULT GETUTCDATE(),
    execution_end DATETIME2(7),
    execution_duration_ms INT,
    
    -- Result tracking
    success BIT DEFAULT 1,
    error_message NVARCHAR(MAX),
    
    FOREIGN KEY (session_id) REFERENCES dbo.chat_sessions(session_id),
    FOREIGN KEY (user_id) REFERENCES dbo.users(id)
);

-- Tool Usage table (enhanced for multi-agent context)
CREATE TABLE dbo.tool_usage (
    tool_call_id NVARCHAR(255) PRIMARY KEY,
    session_id NVARCHAR(255) NOT NULL,
    trace_id NVARCHAR(255),
    tool_id NVARCHAR(255) NOT NULL,
    tool_name NVARCHAR(255) NOT NULL,
    tool_input NVARCHAR(MAX) NOT NULL, -- JSON string
    tool_output NVARCHAR(MAX), -- JSON string
    tool_message NVARCHAR(MAX),
    status NVARCHAR(50) CHECK (status IN ('Success', 'Error', 'Timeout', 'Cancelled')),
    
    -- Multi-agent context
    executing_agent NVARCHAR(255), -- Which agent executed this tool
    agent_type NVARCHAR(100), -- Type of agent that used this tool
    
    -- Performance tracking
    tokens_used INT,
    execution_time_ms INT,
    created_at DATETIME2(7) DEFAULT GETUTCDATE(),
    
    FOREIGN KEY (session_id) REFERENCES dbo.chat_sessions(session_id),
    FOREIGN KEY (trace_id) REFERENCES dbo.chat_history(trace_id),
    FOREIGN KEY (tool_id) REFERENCES dbo.tool_definitions(tool_id)
);

GO

-- View for agent performance analytics
CREATE VIEW dbo.vw_agent_performance AS
SELECT 
    h.agent_type,
    h.agent_name,
    COUNT(h.message_id) as total_messages,
    SUM(CASE WHEN h.message_type = 'ai' THEN h.total_tokens ELSE 0 END) as total_tokens,
    AVG(CASE WHEN h.message_type = 'ai' THEN h.response_time_ms ELSE NULL END) as avg_response_time_ms,
    COUNT(DISTINCT h.session_id) as unique_sessions,
    MIN(h.trace_end) as first_usage,
    MAX(h.trace_end) as last_usage
FROM dbo.chat_history h
WHERE h.agent_type IS NOT NULL
GROUP BY h.agent_type, h.agent_name;

GO

-- View for tool usage analytics
CREATE VIEW dbo.vw_tool_usage_analytics AS
SELECT 
    tu.agent_type,
    tu.executing_agent,
    tu.tool_name,
    COUNT(tu.tool_call_id) as usage_count,
    SUM(CASE WHEN tu.status = 'Success' THEN 1 ELSE 0 END) as success_count,
    SUM(CASE WHEN tu.status = 'Error' THEN 1 ELSE 0 END) as error_count,
    AVG(tu.execution_time_ms) as avg_execution_time_ms,
    SUM(ISNULL(tu.tokens_used, 0)) as total_tokens_used
FROM dbo.tool_usage tu
GROUP BY tu.agent_type, tu.executing_agent, tu.tool_name;

GO

-- View for session analytics
CREATE VIEW dbo.vw_session_analytics AS
SELECT 
    cs.session_id,
    cs.user_id,
    cs.primary_agent_type,
    cs.total_agents_used,
    COUNT(DISTINCT h.agent_type) as unique_agent_types_used,
    COUNT(h.message_id) as total_messages,
    SUM(CASE WHEN h.message_type = 'human' THEN 1 ELSE 0 END) as user_messages,
    SUM(CASE WHEN h.message_type = 'ai' THEN 1 ELSE 0 END) as ai_messages,
    SUM(CASE WHEN h.message_type = 'tool_call' THEN 1 ELSE 0 END) as tool_calls,
    SUM(ISNULL(h.total_tokens, 0)) as total_tokens,
    cs.created_at,
    cs.updated_at,
    DATEDIFF(MINUTE, cs.created_at, cs.updated_at) as session_duration_minutes
FROM dbo.chat_sessions cs
LEFT JOIN dbo.chat_history h ON cs.session_id = h.session_id
GROUP BY cs.session_id, cs.user_id, cs.primary_agent_type, cs.total_agents_used, cs.created_at, cs.updated_at;

GO

-- View for routing analytics
CREATE VIEW dbo.vw_routing_analytics AS
SELECT 
    at.target_agent,
    at.task_type,
    at.coordinator_agent,
    COUNT(at.trace_step_id) as total_routes,
    SUM(CASE WHEN at.success = 1 THEN 1 ELSE 0 END) as successful_routes,
    AVG(at.execution_duration_ms) as avg_execution_duration_ms,
    COUNT(DISTINCT at.session_id) as unique_sessions
FROM dbo.agent_traces at
GROUP BY at.target_agent, at.task_type, at.coordinator_agent;

GO