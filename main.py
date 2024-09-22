import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Use raw string for file path
file_path = r"C:\Users\vishn\OneDrive\Desktop\archive (1)\Parkinsson disease.csv"

# Debug: Starting script
print("Script is starting...")

# Attempt to load data with error handling
try:
    p_data = pd.read_csv(file_path)
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Check if the data is loaded properly
print(p_data.head())

# Prepare the data
a = p_data.drop(columns=['name'])
b = p_data['status']

# Debug: Data preparation
print("Data prepared.")

# Split the dataset into training and test sets
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=42)

# Debug: Starting model training
print("Starting model training...")
classifier = SVC()
classifier.fit(a_train, b_train)
print("Model training completed.")

# Predict on test set
b_pred = classifier.predict(a_test)

# Output classification report and accuracy
print("Classification Report for dataset is:")
print(classification_report(b_test, b_pred))
accuracy = accuracy_score(b_test, b_pred)
print("Accuracy in decimals is:", accuracy)
print("Accuracy percentage is=", accuracy * 100)

# Predict on a new instance
new_instance = pd.DataFrame([[119.992, 157.302, 74.997, 0.00784, 0.00007, 0.0037,
                              0.00554, 0.01109, 0.04374, 0.426, 0.02182, 0.0313,
                              0.02971, 0.06545, 0.02211, 21.033, 1, 0.414783, 
                              0.815285, -4.813031, 0.266482, 2.301442, 0.284654]])

new_instance = new_instance.astype(float)
predict = classifier.predict(new_instance)

# Interpretation of prediction
if predict[0] == 1:
    print("It is better to consult the doctor because the person may have Parkinson's disease.")
else:
    print("The person is prognosticated not to have Parkinson's complaint.")
