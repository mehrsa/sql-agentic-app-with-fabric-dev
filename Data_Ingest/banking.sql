-- Drop tables if they exist to start from a clean slate
IF OBJECT_ID('transactions', 'U') IS NOT NULL DROP TABLE transactions;
IF OBJECT_ID('accounts', 'U') IS NOT NULL DROP TABLE accounts;
IF OBJECT_ID('users', 'U') IS NOT NULL DROP TABLE users;

-- Create the 'users' table
CREATE TABLE users (
    id NVARCHAR(255) PRIMARY KEY,
    name NVARCHAR(255) NOT NULL,
    email NVARCHAR(255) UNIQUE NOT NULL,
    created_at DATETIMEOFFSET DEFAULT SYSDATETIMEOFFSET()
);

-- Create the 'accounts' table
CREATE TABLE accounts (
    id NVARCHAR(255) PRIMARY KEY,
    user_id NVARCHAR(255) FOREIGN KEY REFERENCES users(id),
    account_number NVARCHAR(255) UNIQUE NOT NULL,
    account_type NVARCHAR(255) NOT NULL,
    balance DECIMAL(15, 2) NOT NULL,
    name NVARCHAR(255) NOT NULL,
    created_at DATETIMEOFFSET DEFAULT SYSDATETIMEOFFSET()
);

-- Create the 'transactions' table
CREATE TABLE transactions (
    id NVARCHAR(255) PRIMARY KEY,
    from_account_id NVARCHAR(255) FOREIGN KEY REFERENCES accounts(id),
    to_account_id NVARCHAR(255) FOREIGN KEY REFERENCES accounts(id),
    amount DECIMAL(15, 2) NOT NULL,
    type NVARCHAR(255) NOT NULL,
    description NTEXT,
    category NVARCHAR(255),
    status NVARCHAR(255) NOT NULL,
    created_at DATETIMEOFFSET DEFAULT SYSDATETIMEOFFSET()
);

-- Insert data into the 'users' table
INSERT INTO users (id, name, email, created_at) VALUES
('user_1', 'John Doe', 'john.doe@example.com', '2025-06-24T02:44:13.180Z'),
('user_2', 'Jane Smith', 'jane.smith@example.com', '2025-06-24T02:44:13.180Z'),
('user_3', 'Alice Johnson', 'alice.johnson@example.com', '2025-06-24T02:44:13.180Z'),
('user_4', 'Bob Brown', 'bob.brown@example.com', '2025-06-24T02:44:13.180Z'),
('user_5', 'Charlie Davis', 'charlie.davis@example.com', '2025-06-24T02:44:13.180Z'),
('user_6', 'Diana Evans', 'diana.evans@example.com', '2025-06-24T02:44:13.180Z');


-- Insert data into the 'accounts' table
INSERT INTO accounts (id, user_id, account_number, account_type, balance, name, created_at) VALUES
('acc_1', 'user_1', '123456789', 'checking', 4822.50, 'Primary Checking', '2025-06-24T02:44:13.197Z'),
('acc_2', 'user_1', '987654321', 'savings', 13015.00, 'High-Yield Savings', '2025-06-24T02:44:13.197Z'),
('acc_3', 'user_1', '112233445', 'credit', -490.00, 'Platinum Credit Card', '2025-06-24T02:44:13.197Z'),
('acc_4', 'user_2', '223344556', 'checking', 3050.75, 'Everyday Checking', '2025-06-24T02:44:13.197Z'),
('acc_5', 'user_2', '665544332', 'savings', 8450.00, 'Vacation Savings', '2025-06-24T02:44:13.197Z'),
('acc_6', 'user_3', '334455667', 'checking', 1500.00, 'Student Checking', '2025-06-24T02:44:13.197Z'),
('acc_7', 'user_3', '778899001', 'credit', -1200.00, 'Student Credit Card', '2025-06-24T02:44:13.197Z'),
('acc_8', 'user_4', '445566778', 'checking', 7200.00, 'Joint Checking', '2025-06-24T02:44:13.197Z'),
('acc_9', 'user_5', '556677889', 'savings', 5600.00, 'Emergency Fund', '2025-06-24T02:44:13.197Z'),
('acc_10', 'user_6', '667788990', 'checking', 4300.25, 'Business Checking', '2025-06-24T02:44:13.197Z'),
('acc_11', 'user_6', '998877665', 'credit', -750.00, 'Business Credit Card', '2025-06-24T02:44:13.197Z'),
('acc_12', 'user_5', '776655443', 'checking', 2200.00, 'Secondary Checking', '2025-06-24T02:44:13.197Z');

-- Insert data into the 'transactions' table
INSERT INTO transactions (id, from_account_id, to_account_id, amount, type, description, category, status, created_at) VALUES
('txn_1', 'acc_1', NULL, 75.00, 'payment', 'Grocery Store', 'Groceries', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_2', 'acc_1', NULL, 1200.00, 'payment', 'Rent Payment', 'Housing', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_3', NULL, 'acc_1', 3000.00, 'deposit', 'Paycheck', 'Income', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_4', 'acc_1', 'acc_2', 500.00, 'transfer', 'Monthly Savings', 'Transfer', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_5', 'acc_3', NULL, 150.00, 'payment', 'Dinner with Friends', 'Food', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_6', 'acc_1', NULL, 25.50, 'payment', 'Coffee Shop', 'Food', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_7', 'acc_2', NULL, 200.00, 'payment', 'Electronics Store', 'Shopping', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_8', NULL, 'acc_2', 1500.00, 'deposit', 'Freelance Project', 'Income', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_9', 'acc_4', NULL, 60.00, 'payment', 'Gas Station', 'Transportation', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_10', 'acc_4', NULL, 300.00, 'payment', 'Utility Bill', 'Utilities', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_11', NULL, 'acc_4', 2500.00, 'deposit', 'Salary Deposit', 'Income', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_12', 'acc_5', NULL, 100.00, 'payment', 'Bookstore', 'Education', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_13', NULL, 'acc_5', 800.00, 'deposit', 'Tax Refund', 'Income', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_14', 'acc_6', NULL, 45.00, 'payment', 'Fast Food Restaurant', 'Food', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_15', NULL, 'acc_6', 1200.00, 'deposit', 'Parent Gift', 'Income', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_16', 'acc_7', NULL, 300.00, 'payment', 'Online Shopping', 'Shopping', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_17', NULL, 'acc_7', 500.00, 'deposit', 'Scholarship', 'Income', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_18', 'acc_8', NULL, 400.00, 'payment', 'Home Improvement Store', 'Home', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_19', NULL, 'acc_8', 3500.00, 'deposit', 'Bonus Payment', 'Income', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_20', 'acc_9', NULL, 75.00, 'payment', 'Pharmacy', 'Health', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_21', NULL, 'acc_9', 600.00, 'deposit', 'Gift from Friend', 'Income', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_22', 'acc_10', NULL, 250.00, 'payment', 'Office Supplies Store', 'Business Expenses', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_23', NULL, 'acc_10', 4000.00, 'deposit', 'Client Payment', 'Income', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_24', 'acc_11', NULL, 100.00, 'payment', 'Business Lunch', 'Business Expenses', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_25', NULL, 'acc_11', 1200.00, 'deposit', 'Investor Funding', 'Income', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_26', 'acc_12', NULL, 60.00, 'payment', 'Grocery Store', 'Groceries', 'completed', '2025-06-24T02:44:13.217Z'),
('txn_27', NULL, 'acc_12', 700.00, 'deposit', 'Side Business Income', 'Income', 'completed', '2025-06-24T02:44:13.217Z');
-- Note: This is a sample dataset for a banking application. The transactions include various types such as payments, deposits, and transfers across different accounts.
