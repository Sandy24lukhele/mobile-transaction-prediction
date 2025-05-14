from flask import Flask, request, render_template, redirect, url_for, session
import numpy as np
import joblib
import os
import datetime
import mysql.connector
from decimal import Decimal

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Load models and encoders
model_rf = joblib.load(os.path.join('models', 'Random Forest_model.pkl'))
model_svm = joblib.load(os.path.join('models', 'SVM_model.pkl'))
model_gb = joblib.load(os.path.join('models', 'Gradient Boosting_model.pkl'))
model_ann = joblib.load(os.path.join('models', 'ANN_model.pkl'))
model_lr = joblib.load(os.path.join('models', 'Logistic Regression_model.pkl'))
label_encoders = joblib.load(os.path.join('models', 'label_encoders.pkl'))
scaler = joblib.load(os.path.join('models', 'scaler.pkl'))

# Database functions
def get_sender_balance(phone_number):
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='19425Owen.',
        database='mobile_money_db'
    )
    cursor = conn.cursor()
    cursor.execute("SELECT balance FROM users WHERE phone_number = %s", (phone_number,))
    balance = cursor.fetchone()
    conn.close()
    return balance[0] if balance else None

def update_sender_balance(phone_number, new_balance):
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='19425Owen.',
        database='mobile_money_db'
    )
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET balance = %s WHERE phone_number = %s", (new_balance, phone_number))
    conn.commit()
    conn.close()

def record_transaction(user_id, phone_number, transaction_type, amount, sender_balance_before, sender_balance_after, time_of_transaction, day_of_week, status, failure_reason, network_provider, prediction_success_probability, prediction_failure_probability):
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='19425Owen.',
        database='mobile_money_db'
    )
    cursor = conn.cursor()
    cursor.execute(""" 
        INSERT INTO transactions (user_id, phone_number, transaction_type, amount, sender_balance_before, sender_balance_after, time_of_transaction, day_of_week, status, failure_reason, network_provider, prediction_success_probability, prediction_failure_probability) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
    """, (user_id, phone_number, transaction_type, amount, sender_balance_before, sender_balance_after, time_of_transaction, day_of_week, status, failure_reason, network_provider, prediction_success_probability, prediction_failure_probability))
    conn.commit()
    conn.close()

def predict_transaction_status(data):
    network_provider = label_encoders['Network_Provider'].transform([data['network_provider']])
    day_of_week = label_encoders['Day_of_Week'].transform([data['day_of_week']])
    transaction_type = label_encoders['Transaction_Type'].transform([data['transaction_type']])

    input_features = np.array([[network_provider[0], day_of_week[0], transaction_type[0],
                                data['amount'], data['sender_balance'], data['time_of_day']]])
    scaled_input = scaler.transform(input_features)

    rf_pred = model_rf.predict_proba(scaled_input)[0][1]
    svm_pred = model_svm.predict_proba(scaled_input)[0][1]
    gb_pred = model_gb.predict_proba(scaled_input)[0][1]
    ann_pred = model_ann.predict_proba(scaled_input)[0][1]
    lr_pred = model_lr.predict_proba(scaled_input)[0][1]

    avg_pred = np.mean([rf_pred, svm_pred, gb_pred, ann_pred, lr_pred])
    failure_pred = 1 - avg_pred

    # Control formatting for small probabilities
    failure_pred_display = f"{failure_pred * 100:.2f}"  # Shows two decimal places

    if failure_pred > avg_pred:
        message = f"This is most likely to be a Failed transaction with {failure_pred_display}% probability."
        return 'Failed', message, avg_pred, failure_pred
    else:
        message = f"This is most likely to be a Successful transaction with {avg_pred * 100:.2f}% probability."
        return 'Success', message, avg_pred, failure_pred  # Return failure_pred for both success and failure

# Routes
@app.route('/')
def index():
    if 'phone_number' not in session:
        return redirect(url_for('login'))

    current_time = datetime.datetime.now()
    time_of_day = current_time.hour
    day_of_week = current_time.strftime('%A')

    phone_number = session['phone_number']
    current_balance = get_sender_balance(phone_number)

    return render_template('index.html', time_of_day=time_of_day, day_of_week=day_of_week, current_balance=current_balance)

@app.route('/predict', methods=['POST'])
def predict():
    if 'phone_number' not in session:
        return redirect(url_for('login'))

    try:
        current_time = datetime.datetime.now()
        time_of_day = current_time.hour
        day_of_week = current_time.strftime('%A')

        data = request.form
        phone_number = session['phone_number']

        sender_balance = get_sender_balance(phone_number)
        if sender_balance is None:
            return render_template('index.html', error="Sender balance not found.", time_of_day=time_of_day, day_of_week=day_of_week)

        transaction_data = {
            'network_provider': data['network_provider'],
            'transaction_type': data['transaction_type'],
            'amount': float(data['amount']),
            'sender_balance': sender_balance,
            'time_of_day': time_of_day,
            'day_of_week': day_of_week
        }

        # Predict transaction status
        status, message, success_prob, failure_prob = predict_transaction_status(transaction_data)

        # Display prediction result
        return render_template('index.html', predictions={'Message': message, 'Probability of Success': f'{success_prob * 100:.2f}%', 'Probability of Failure': f'{failure_prob * 100:.2f}%'}, 
                               time_of_day=time_of_day, day_of_week=day_of_week, current_balance=sender_balance, status=status, 
                               amount=data['amount'], transaction_type=data['transaction_type'], network_provider=data['network_provider'])

    except Exception as e:
        return render_template('index.html', error=str(e), time_of_day=time_of_day, day_of_week=day_of_week)

@app.route('/complete_transaction', methods=['POST'])
def complete_transaction():
    if 'phone_number' not in session:
        return redirect(url_for('login'))

    try:
        current_time = datetime.datetime.now()
        time_of_day = current_time.hour
        day_of_week = current_time.strftime('%A')

        pin = request.form['pin']
        transaction_type = request.form['transaction_type']
        amount = float(request.form['amount'])
        network_provider = request.form['network_provider']

        # Convert amount and sender balance to Decimal
        amount = Decimal(str(amount))

        # Get the status from the prediction (Success or Failed)
        status = request.form['status'].strip().capitalize()  # Predict status from the models (e.g., Success/Failed)
        
        # Validate the PIN
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='19425Owen.',
            database='mobile_money_db'
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE phone_number = %s AND pin = %s", (session['phone_number'], pin))
        user = cursor.fetchone()
        conn.close()

        if user:
            sender_balance = get_sender_balance(session['phone_number'])

            # Convert sender_balance to Decimal
            sender_balance = Decimal(str(sender_balance))

            if sender_balance is None:
                return render_template('index.html', error="Sender balance not found.", time_of_day=time_of_day, day_of_week=day_of_week)

            if sender_balance < amount:
                # Insufficient balance
                status = 'Failed'
                failure_reason = 'Insufficient balance'
                sender_balance_after = sender_balance  # Balance remains unchanged
            else:
                # Perform prediction to determine if the transaction is likely to succeed or fail
                status, message, success_prob, failure_prob = predict_transaction_status({
                    'amount': amount,
                    'sender_balance': sender_balance,
                    'time_of_day': time_of_day,
                    'day_of_week': day_of_week,
                    'network_provider': network_provider,
                    'transaction_type': transaction_type
                })

                if status == 'Success':
                    # Deduct the amount for successful transaction
                    sender_balance_after = sender_balance - amount
                else:
                    # In case of failure, balance remains the same
                    sender_balance_after = sender_balance
                    failure_reason = message  # Include prediction message for failed transactions

            # Update sender's balance in the database
            update_sender_balance(session['phone_number'], sender_balance_after)

            # Record the transaction with the outcome
            record_transaction(
                user_id=user['id'],
                phone_number=session['phone_number'],
                transaction_type=transaction_type,
                amount=amount,
                sender_balance_before=sender_balance,
                sender_balance_after=sender_balance_after,
                time_of_transaction=current_time.strftime('%H:%M:%S'),
                day_of_week=day_of_week,
                status=status,
                failure_reason=failure_reason if status == 'Failed' else None,
                network_provider=network_provider,
                prediction_success_probability=success_prob if status == 'Success' else None,
                prediction_failure_probability=failure_prob if status == 'Failed' else 0.0  # Explicitly set 0.0 for 'Success' status
            )

            # Provide feedback to the user based on the transaction outcome
            if status == 'Success':
                return render_template('index.html', success="Transaction completed successfully", time_of_day=time_of_day, day_of_week=day_of_week, current_balance=sender_balance_after)
            else:
                return render_template('index.html', error=f"Transaction failed: {failure_reason}", time_of_day=time_of_day, day_of_week=day_of_week)

        else:
            return render_template('index.html', error="Incorrect PIN, transaction not completed.", time_of_day=time_of_day, day_of_week=day_of_week)

    except Exception as e:
        return render_template('index.html', error=str(e), time_of_day=time_of_day, day_of_week=day_of_week)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        number = request.form['phone_number']
        password = request.form['password']

        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='19425Owen.',
            database='mobile_money_db'
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE phone_number = %s AND pin = %s", (number, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session['user_id'] = user['id']
            session['phone_number'] = number
            return redirect(url_for('index'))

        return render_template('login.html', error="Invalid credentials, please try again.")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)

