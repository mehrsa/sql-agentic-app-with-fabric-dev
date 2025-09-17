import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User } from 'lucide-react';

// Simplified ChatMessage interface for the frontend
interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface ChatBotProps {
  // The backend now handles all logic, so no props are needed
}

const ChatBot: React.FC<ChatBotProps> = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content: "Hello! I'm your AI banking assistant. I can help you check balances, transfer funds, create new accounts, and analyze your spending. How can I assist you today?",
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const API_URL = 'http://127.0.0.1:5001/api';
  const ANALYTICS_API_URL = 'http://127.0.0.1:5002/api';

  // Always create a new session on component mount (ignoring localStorage)
  useEffect(() => {
    const createNewSession = async () => {
      try {
        const response = await fetch(`${ANALYTICS_API_URL}/chat/sessions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            title: `Chat Session ${new Date().toLocaleString()}`,
            user_id: 'user_1' 
          }),
        });
        
        if (response.ok) {
          const data = await response.json();
          setSessionId(data.session_id);
          localStorage.setItem('chatSessionId', data.session_id);
          console.log(`✅ New chat session created: ${data.session_id}`);
        }
      } catch (error) {
        console.error("Failed to create session:", error);
        // If session creation fails, generate a temporary one with timestamp
        const tempSessionId = `temp_session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        setSessionId(tempSessionId);
        localStorage.setItem('chatSessionId', tempSessionId);
        console.log(`🔄 Fallback session created: ${tempSessionId}`);
      }
    };

    // Always create a new session, regardless of what's in localStorage
    createNewSession();
  }, []); // Empty dependency array ensures this runs only once on mount

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || !sessionId) return;

    const newMessages: ChatMessage[] = [...messages, { role: 'user', content: inputMessage }];
    setMessages(newMessages);
    setInputMessage('');
    setIsTyping(true);

    try {
      // Send both messages and session_id to the backend
      const response = await fetch(`${API_URL}/chatbot`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          messages: newMessages,
          session_id: sessionId,
          user_id: 'user_1'
        }), 
      });

      if (!response.ok) {
        throw new Error('Failed to get response from the assistant.');
      }

      const data = await response.json();
      
      // Update session_id if the backend returns a new one
      if (data.session_id && data.session_id !== sessionId) {
        setSessionId(data.session_id);
        localStorage.setItem('chatSessionId', data.session_id);
      }
      
      // Add the AI's final response to the history
      setMessages([...newMessages, { role: 'assistant', content: data.response }]);

    } catch (error) {
      console.error("Chatbot error:", error);
      setMessages([...newMessages, { role: 'assistant', content: "Sorry, I'm having trouble connecting to my brain right now. Please try again later." }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Function to start a new chat session
  const startNewSession = async () => {
    try {
      const response = await fetch(`${ANALYTICS_API_URL}/chat/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          title: `New Chat ${new Date().toLocaleString()}`,
          user_id: 'user_1' 
        }),
      });
      
      if (response.ok) {
        const data = await response.json();
        setSessionId(data.session_id);
        localStorage.setItem('chatSessionId', data.session_id);
        
        // Reset messages to initial state
        setMessages([{
          role: "assistant",
          content: "Hello! I'm your AI banking assistant. I can help you check balances, transfer funds, create new accounts, and analyze your spending. How can I assist you today?",
        }]);
      }
    } catch (error) {
      console.error("Failed to create new session:", error);
    }
  };

  return (
    <div className="flex flex-col h-full bg-white">
      <div className="bg-gradient-to-r from-blue-600 to-blue-700 text-white p-4 rounded-t-xl flex justify-between items-center">
        <div>
          <h3 className="font-semibold">AI Banking Assistant</h3>
          {sessionId && (
            <p className="text-xs text-blue-100">Session: {sessionId.substring(0, 8)}...</p>
          )}
        </div>
        <button 
          onClick={startNewSession}
          className="px-3 py-1 bg-blue-500 hover:bg-blue-400 rounded text-xs"
        >
          New Chat
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message, index) => (
          <div key={index} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`px-4 py-2 rounded-2xl ${message.role === 'user' ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-800'}`}>
              <pre className="whitespace-pre-wrap font-sans text-sm">{message.content}</pre>
            </div>
          </div>
        ))}
        {isTyping && (
          <div className="flex justify-start">
            <div className="px-4 py-2 rounded-2xl bg-gray-100">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="p-4 border-t">
        <div className="flex space-x-2">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me anything..."
            className="flex-1 px-4 py-2 border rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={!sessionId}
          />
          <button 
            onClick={handleSendMessage} 
            disabled={isTyping || !sessionId} 
            className="px-4 py-2 bg-blue-600 text-white rounded-full hover:bg-blue-700 disabled:opacity-50"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatBot;