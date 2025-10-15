-- Use this to create the core chat history tables:

-- 1. Chat Sessions Table to manage individual conversation sessions
CREATE TABLE chat_sessions (
    session_id NVARCHAR(255) PRIMARY KEY,
    user_id NVARCHAR(255) NOT NULL,
    title NVARCHAR(500),
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    updated_at DATETIME2 DEFAULT GETUTCDATE(),
    duration_seconds AS DATEDIFF(SECOND, created_at, updated_at) PERSISTED
);

-- 2. Agent Definitions Table 
CREATE TABLE agent_definitions (
    agent_id NVARCHAR(255) PRIMARY KEY,
    name NVARCHAR(255) UNIQUE NOT NULL,  
    description NVARCHAR(MAX),
    llm_config NVARCHAR(MAX) NOT NULL,   -- This should be JSON type if supported, otherwise NVARCHAR(MAX)
    prompt_template NVARCHAR(MAX) NOT NULL
);

-- 3. Tool Definitions Table (moved up before other tables that reference it)
CREATE TABLE tool_definitions (
    tool_id NVARCHAR(255) PRIMARY KEY,  -- Changed from 'id' to 'tool_id'
    name NVARCHAR(255) UNIQUE NOT NULL,
    description NVARCHAR(MAX),
    input_schema NVARCHAR(MAX) NOT NULL, -- JSON Schema for input validation
    version NVARCHAR(50) DEFAULT '1.0.0',
    is_active BIT DEFAULT 1,
    cost_per_call_cents INT DEFAULT 0,
    created_at DATETIME2 DEFAULT GETUTCDATE(),
    updated_at DATETIME2 DEFAULT GETUTCDATE()
);


-- 4. Chat History Table for storing all conversation messages and tool interactions
CREATE TABLE chat_history (
    message_id NVARCHAR(255) PRIMARY KEY,
    session_id NVARCHAR(255) NOT NULL,
    trace_id NVARCHAR(255) NOT NULL,     -- Add trace_id field
    user_id NVARCHAR(255) NOT NULL,
    agent_id NVARCHAR(255),              -- Add agent_id field
    message_type NVARCHAR(50) NOT NULL,  -- 'human', 'ai', 'system', 'tool_call', 'tool_result'
    content NVARCHAR(MAX),
 
    model_name NVARCHAR(255),          -- Add model_name field
    content_filter_results NVARCHAR(MAX) DEFAULT '{}', -- JSON data for content filter results
    total_tokens INT,                   -- Total tokens in the message
    completion_tokens INT,              -- Tokens used for completion
    prompt_tokens INT,                  -- Tokens used for the prompt

    finish_reason NVARCHAR(255),       -- Reason for finishing the message
    response_time_ms INT,              -- Response time in milliseconds
    trace_end DATETIME2 DEFAULT GETUTCDATE(),           -- End time of the trace

    -- Tool usage fields (embedded in message)
    tool_call_id NVARCHAR(255),
    tool_name NVARCHAR(255),
    tool_input NVARCHAR(MAX),            -- JSON data for tool input parameters
    tool_output NVARCHAR(MAX),           -- JSON data for tool output/results
    tool_id NVARCHAR(255)               -- Add tool_id field

);


-- 5. Tool Usage Table for detailed metrics and tracking for individual tool executions
CREATE TABLE tool_usage (
    tool_call_id NVARCHAR(255) PRIMARY KEY,  -- Changed from 'id' to 'tool_call_id' to match Python model
    session_id NVARCHAR(255) NOT NULL,
    trace_id NVARCHAR(255),              -- Add trace_id field
    tool_id NVARCHAR(255) NOT NULL,      -- Add tool_id field (foreign key to tool_definitions)
    tool_name NVARCHAR(255) NOT NULL,
    tool_input NVARCHAR(MAX) NOT NULL,   -- JSON data for input parameters
    tool_output NVARCHAR(MAX),
    tool_message NVARCHAR(MAX),
    status NVARCHAR(50),
    tokens_used INT                     -- Number of tokens consumed
);


-- Foreign Key Constraints
-- Link chat_history to chat_sessions
-- ALTER TABLE chat_history 
-- ADD CONSTRAINT FK_chat_history_session 
-- FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id);
