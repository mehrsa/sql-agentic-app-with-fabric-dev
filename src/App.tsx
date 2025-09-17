import React, { useState, useEffect } from 'react';
import Layout from './components/Layout';
import Dashboard from './components/Dashboard';
import Transactions from './components/Transactions';
import Transfer from './components/Transfer';
import Analytics from './components/Analytics';
import ChatBot from './components/ChatBot';
import { MessageCircle, X } from 'lucide-react';

import ChatSessions from './components/ChatSessions';
import ToolAnalytics from './components/ToolAnalytics';

import type { Account, Transaction } from './types/banking';

const API_URL = 'http://127.0.0.1:5001/api';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Clear previous session data on app launch
    localStorage.removeItem('chatSessionId');
    console.log('🧹 Previous chat session cleared - new session will be created');
    
    loadBankingData();
  }, []);
  
  const loadBankingData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [accountsResponse, transactionsResponse] = await Promise.all([
        fetch(`${API_URL}/accounts`),
        fetch(`${API_URL}/transactions`),
      ]);
      if (!accountsResponse.ok || !transactionsResponse.ok) {
        throw new Error('Failed to fetch data from the server.');
      }
      const accountsData = await accountsResponse.json();
      const transactionsData = await transactionsResponse.json();
      setAccounts(accountsData);
      setTransactions(transactionsData);
    } catch (error) {
      console.error('Error loading banking data:', error);
      setError('Could not connect to the banking service. Please ensure the backend is running and refresh.');
    } finally {
      setLoading(false);
    }
  };

  // This function now just triggers a refresh, as the chatbot handles the transaction logic.
 const handleTransactionComplete = async (transactionData: Omit<Transaction, 'id' | 'created_at' | 'status'>, fromAccountName: string, toAccountName?: string) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_URL}/transactions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          from_account_name: fromAccountName,
          to_account_name: toAccountName,
          amount: transactionData.amount,
          description: transactionData.description
        }),
      });

      if (!response.ok) {
        const errorResult = await response.json();
        throw new Error(errorResult.message || 'Failed to complete transaction.');
      }
      
      // Transaction successful, now reload all data
      await loadBankingData();

    } catch (error: any) {
      console.error('Error completing transaction:', error);
      setError(error.message || 'An unexpected error occurred during the transaction.');
    } finally {
      setLoading(false);
    }
  };

  // New function to handle account creation from the Transfer component
  const handleAccountCreate = async (accountData: { account_type: 'checking' | 'savings', name: string, balance: number }) => {
    try {
      const response = await fetch(`${API_URL}/accounts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(accountData),
      });
      if (!response.ok) throw new Error('Account creation failed.');
      const newAccount = await response.json();
      // Refresh all data to get the new account list
      await loadBankingData();
      return newAccount; // Return new account details if needed
    } catch (err) {
      console.error("Error creating account:", err);
      setError("There was an error creating the new account.");
      throw err; // Re-throw to be caught by the calling component
    }
  };


  const renderContent = () => {
    if (loading) return <div className="text-center p-8">Loading Banking Data...</div>;
    if (error) return <div className="text-center p-8 text-red-600 bg-red-50 rounded-lg">{error}</div>;

    switch (activeTab) {
      case 'dashboard':
        return <Dashboard accounts={accounts} recentTransactions={transactions} />;
      case 'transactions':
        return <Transactions transactions={transactions} accounts={accounts} />;
      case 'transfer':
        // Pass both handlers to the Transfer component
        return <Transfer accounts={accounts} onTransactionComplete={handleTransactionComplete} onAccountCreate={handleAccountCreate} />;
      case 'analytics':
        return <Analytics transactions={transactions} accounts={accounts} />;
      
      case 'chat-sessions':
        return <ChatSessions />;
      case 'tool-analytics':
        return <ToolAnalytics />;
        
      default:
        return <Dashboard accounts={accounts} recentTransactions={transactions} />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Layout activeTab={activeTab} onTabChange={setActiveTab}>{renderContent()}</Layout>
      <button onClick={() => setIsChatOpen(!isChatOpen)} className={`fixed bottom-6 right-6 p-4 rounded-full shadow-lg transition-all z-40 ${isChatOpen ? 'bg-red-600' : 'bg-blue-600'} text-white`}>
        {isChatOpen ? <X/> : <MessageCircle/>}
      </button>
      {isChatOpen && (
        <div className="fixed bottom-24 right-6 w-96 h-[500px] bg-white rounded-xl shadow-2xl border z-30">
          <ChatBot />
        </div>
      )}
    </div>
  );
}

export default App;