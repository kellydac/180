# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

# Load the dataset from Excel file
file_path = r'C:\Users\kelly\Downloads\baseball.xlsx'
baseball_data = pd.read_excel(file_path)

# Select relevant columns for prediction and target variable
# 'RS' stands for Runs Scored, 'RA' stands for Runs Allowed,
# 'W' stands for Wins, 'OBP' stands for On-Base Percentage,
# 'SLG' stands for Slugging Percentage, 'BA' stands for Batting Average
# 'Playoffs' is the target variable indicating if a team made it to playoffs (1) or not (0)
X = baseball_data[['Runs Scored', 'Runs Allowed', 'Wins', 'OBP', 'SLG', 'Team Batting Average']]
y = baseball_data['Playoffs']

# Initialize the Logistic Regression model with increased max_iter
model = LogisticRegression(max_iter=1000)

# Train the model on the entire dataset
model.fit(X, y)

# Make predictions on the training data to calculate accuracy
predictions = model.predict(X)
accuracy = accuracy_score(y, predictions)
print("Accuracy of the prediction model:", accuracy)

# Suppress the warning about feature names
warnings.filterwarnings("ignore", category=UserWarning)

# Function to predict playoff probability for a given team
def predict_playoffs(team_name):
    # Find the row corresponding to the team name
    team_row = baseball_data[baseball_data['Team'] == team_name]
    # Extract team data from the row
    team_data = team_row[['Runs Scored', 'Runs Allowed', 'Wins', 'OBP', 'SLG', 'Team Batting Average']].values
    # Predict playoff probability
    new_probabilities = model.predict_proba(team_data)
    return new_probabilities

# Function to input team name and get prediction
def get_prediction():
    team_name = input("Enter the team name: ")
    probabilities = predict_playoffs(team_name)
    playoff_probability = probabilities[0][1]
    if playoff_probability >= 0.5:
        print(team_name, "is predicted to make the playoffs.")
    else:
        print(team_name, "is predicted not to make the playoffs.")
    print("Probability of", team_name, "making the playoffs:", playoff_probability)

# Get prediction for a specific team
get_prediction()

print("Go Brewers") 