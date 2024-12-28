import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = "data.csv"
data = pd.read_csv(file_path)

# Preprocessing the data
# Convert elapsed_time to seconds
def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

data['elapsed_time_seconds'] = data['elapsed_time'].apply(time_to_seconds)

# Encode categorical variables
categorical_columns = ['player1', 'player2', 'p1_score', 'p2_score', 'winner_shot_type', 'serve_width', 'serve_depth', 'return_depth']
label_encoders = {col: LabelEncoder() for col in categorical_columns}
for col in categorical_columns:
    data[col] = label_encoders[col].fit_transform(data[col].astype(str))

# Fill missing numeric values with the median
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

# Define features for point-level prediction
point_features = [
    'elapsed_time_seconds', 'set_no', 'game_no', 'point_no', 'server',
    'p1_sets', 'p2_sets', 'p1_games', 'p2_games',
    'p1_points_won', 'p2_points_won', 'p1_break_pt', 'p2_break_pt'
]

# Target: Who won the point (binary classification)
data['point_winner'] = (data['p1_points_won'] > data['p2_points_won']).astype(int)

# Define features and target for the whole dataset
X_points_full = data[point_features]
y_points_full = data['point_winner']

# Train Gradient Boosting Classifier on the entire dataset
gbr_full = GradientBoostingClassifier()
gbr_full.fit(X_points_full, y_points_full)

# Feature importance for the whole dataset
point_feature_importance_full = pd.DataFrame({
    'Feature': point_features,
    'Importance': gbr_full.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Display the feature importance for the whole dataset
print("Point-level Feature Importance for the Entire Dataset:\n", point_feature_importance_full)

# Use the model to predict the probability of Player 1 winning a point
data['p1_point_probability'] = gbr_full.predict_proba(X_points_full)[:, 1]

# Match-level prediction (existing logic)
def calculate_match_probability(row):
    total_sets = 3 if row['set_no'] <= 3 else 5  # Best of 3 or 5 sets
    sets_to_win = total_sets // 2 + 1
  
    # Weight sets more heavily
    set_score_diff = row['p1_sets'] - row['p2_sets']
    set_momentum = (row['p1_sets'] / sets_to_win) - (row['p2_sets'] / sets_to_win)
  
    # Weight games moderately
    game_score_diff = row['p1_games'] - row['p2_games']
    game_momentum = game_score_diff / 6  # Normalize by max games in a set
  
    # Points have the least weight
    point_score_diff = row['p1_points_won'] - row['p2_points_won']
    point_momentum = row['p1_point_probability']  # Using predicted point probability
  
    return 0.5 + set_momentum * 0.6 + game_momentum * 0.3 + point_momentum * 0.1

data['p1_momentum'] = data.apply(calculate_match_probability, axis=1)
data['p2_momentum'] = 1 - data['p1_momentum']

# Visualization with red vertical lines at the end of each set
def plot_momentum_with_sets(data_segment, title):
    plt.figure(figsize=(12, 6))
    plt.plot(data_segment['elapsed_time_seconds'], data_segment['p1_momentum'], label="Player 1 Momentum", linewidth=2)
    plt.plot(data_segment['elapsed_time_seconds'], data_segment['p2_momentum'], label="Player 2 Momentum", linewidth=2, linestyle="--")

    # Add red vertical lines for the end of each set
    set_changes = data_segment[data_segment['set_no'].diff() > 0]
    for _, row in set_changes.iterrows():
        plt.axvline(x=row['elapsed_time_seconds'], color='red', linestyle='--', linewidth=1, label='Set Change')

    plt.xlabel("Elapsed Time (seconds)", fontsize=12)
    plt.ylabel("Momentum (Probability of Winning the Match)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True)
    plt.show()

# Plot for first and last matches
first_match = data.head(301)
last_match = data.iloc[6952:7286]

plot_momentum_with_sets(first_match, "Momentum vs. Time with Set Changes (First Match)")
plot_momentum_with_sets(last_match, "Momentum vs. Time with Set Changes (Last Match)")
