import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Load dataset
file_path = "/content/Mobile_Money_Transactions_MTN.csv"
df = pd.read_csv(file_path)

# Handle missing values
df.fillna("Unknown", inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Network_Provider', 'Day_of_Week', 'Transaction_Type', 'Status']
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Convert Time_of_Day to hour
df['Time_of_Day'] = pd.to_datetime(df['Time_of_Day'], format='%H:%M').dt.hour

# Feature selection
features = ['Network_Provider', 'Day_of_Week', 'Transaction_Type', 'Amount', 'Sender_Balance', 'Time_of_Day']
X = df[features]
y = df['Status']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42),  # Enable probability
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "ANN": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42)
}

# Train models and evaluate performance
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    # Classification report
    class_report = classification_report(y_test, y_pred)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # AUC (if the model supports probability output)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = "N/A"

    # Print results for each model
    print(f"{name} Evaluation Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Classification Report:\n{class_report}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"AUC Score: {auc}\n")


import joblib  # To save the models

# Save each model to disk
for name, model in models.items():
    joblib.dump(model, f"{name}_model.pkl")
    print(f"Model {name} saved!")

# Optionally, save the label encoders and scaler for consistent preprocessing in the web app
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Label encoders and scaler saved!")

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Assume you have already trained models and preprocessed data

# User input collection (example)
def collect_user_input():
    # Collecting input for the features
    print("Enter the following transaction details:")

    network_provider = input("Network Provider (e.g., MTN, Airtel): ")
    transaction_type = input("Transaction Type (e.g., Send Money, Bill Payment, Airtime Purchase): ")
    amount = float(input("Amount (e.g., 1000): "))
    sender_balance = float(input("Sender Balance (e.g., 5000): "))
    time_of_day = input("Time of Day (HH:MM, e.g., 15:30): ")

    day_of_week = input("Day of Week (e.g., Monday, Tuesday, etc.): ")

    # Returning the collected values as a dictionary
    return {
        'Network_Provider': network_provider,
        'Transaction_Type': transaction_type,
        'Amount': amount,
        'Sender_Balance': sender_balance,
        'Time_of_Day': time_of_day,
        'Day_of_Week': day_of_week
    }

# Preprocessing function for user input
def preprocess_input(user_input, label_encoders, scaler):
    # Encoding categorical variables using the label encoders
    network_provider = label_encoders['Network_Provider'].transform([user_input['Network_Provider']])[0]
    transaction_type = label_encoders['Transaction_Type'].transform([user_input['Transaction_Type']])[0]
    day_of_week = label_encoders['Day_of_Week'].transform([user_input['Day_of_Week']])[0]

    # Convert time_of_day to hour
    time_of_day_hour = pd.to_datetime(user_input['Time_of_Day'], format='%H:%M').hour

    # Creating the feature vector for input
    input_features = np.array([
        network_provider,
        day_of_week,
        transaction_type,
        user_input['Amount'],
        user_input['Sender_Balance'],
        time_of_day_hour
    ]).reshape(1, -1)

    # Scaling the features
    scaled_features = scaler.transform(input_features)
    return scaled_features

# Making predictions using a trained model (assuming Random Forest model is trained)
def predict_status(user_input, model, label_encoders, scaler):
    # Preprocess the input
    preprocessed_input = preprocess_input(user_input, label_encoders, scaler)

    # Get probabilities using the trained model
    prob = model.predict_proba(preprocessed_input)

    # The prob array contains probabilities for both classes [0, 1], where:
    # prob[0] is the probability for 'Failed' (class 0)
    # prob[1] is the probability for 'Success' (class 1)

    prob_failed = prob[0][0] * 100  # Probability for 'Failed' as a percentage
    prob_success = prob[0][1] * 100  # Probability for 'Success' as a percentage

    # Printing the probabilities
    print(f"Probability of Failure: {prob_failed:.2f}%")
    print(f"Probability of Success: {prob_success:.2f}%")

    # Decision on which status is most likely
    if prob_success > prob_failed:
        print(f"This is most likely to be a Success with {prob_success:.2f}% probability.")
    else:
        print(f"This is most likely to be a Failed transaction with {prob_failed:.2f}% probability.")

# Simulating a trained model (for testing purposes)
# You should replace this with your actual trained model
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Use your trained model here
model.fit(np.random.randn(100, 6), np.random.choice([0, 1], size=100))  # Dummy training

# Assuming the following label encoders and scaler have been trained
label_encoders = {
    'Network_Provider': LabelEncoder(),
    'Transaction_Type': LabelEncoder(),
    'Day_of_Week': LabelEncoder()
}

# Fit the label encoders (this is just an example, replace with actual fitting from your dataset)
label_encoders['Network_Provider'].fit(['MTN', 'Airtel'])  # Example fitting
label_encoders['Transaction_Type'].fit(['Send Money', 'Bill Payment', 'Airtime Purchase'])
label_encoders['Day_of_Week'].fit(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Scaling (example)
scaler = StandardScaler()
scaler.fit(np.random.randn(100, 6))  # Example fitting

# Get user input and predict the status
user_input = collect_user_input()
predict_status(user_input, model, label_encoders, scaler)
