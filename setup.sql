CREATE DATABASE IF NOT EXISTS mobile_money_db;
USE mobile_money_db;

CREATE TABLE IF NOT EXISTS users (
    phone_number VARCHAR(20) PRIMARY KEY,
    name VARCHAR(100),
    balance DECIMAL(10, 2)
);

-- Insert dummy users
INSERT INTO users (phone_number, name, balance) VALUES ('0551001001', 'User A', 1200.50);
INSERT INTO users (phone_number, name, balance) VALUES ('0552002002', 'User B', 250.00);
