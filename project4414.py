import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D


data = pandas.read_csv("data.csv")

X = data[['GPA', 'SAT']]
y = data['Rank']

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using all the data
regr.fit(X, y)


print(regr.coef_)

# print(f"Linear Regression Equation: {equation}")

# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the data points
scatter = ax.scatter(X['GPA'], X['SAT'], y, color='blue', marker='o', alpha=0.5)

# Create a custom legend entry for the scatter plot
legend_entry = Line2D([0], [0], marker='o', color='w', label='Actual Data', markerfacecolor='blue', markersize=8)

# Add the legend entry
ax.legend(handles=[legend_entry])

# Create a meshgrid for the plane
gpa_range = np.linspace(min(X['GPA']), max(X['GPA']), 20)
sat_range = np.linspace(min(X['SAT']), max(X['SAT']), 20)
gpa_values, sat_values = np.meshgrid(gpa_range, sat_range)

# Predict the corresponding rank for each point in the meshgrid
rank_values = regr.predict(np.c_[gpa_values.ravel(), sat_values.ravel()])
rank_values = rank_values.reshape(gpa_values.shape)

# Plot the regression plane
ax.plot_surface(gpa_values, sat_values, rank_values, color='red', alpha=0.3, label='Regression Plane')

# Set labels for the axes
ax.set_xlabel('GPA')
ax.set_ylabel('SAT')
ax.set_zlabel('Rank')

# Show the plot
plt.title('3D Linear Regression')
plt.show()

# Take user input for GPA and SAT with validation
while True:
    try:
        user_gpa = float(input("Enter GPA (0-4): "))
        if 0 <= user_gpa <= 4:
            break
        else:
            print("Please enter a GPA between 0 and 4.")
    except ValueError:
        print("Invalid input. Please enter a valid number for GPA.")

while True:
    try:
        user_sat = float(input("Enter SAT score (400-1600): "))
        if 400 <= user_sat <= 1600:
            break
        else:
            print("Please enter an SAT score between 400 and 1600.")
    except ValueError:
        print("Invalid input. Please enter a valid number for SAT score.")

# Make a prediction using the trained model
user_data = np.array([[user_gpa, user_sat]])
predicted_rank = regr.predict(user_data)

# Round the predicted rank to the nearest positive whole number
rounded_rank = round(predicted_rank[0])

# Display the rounded predicted rank
print(f"Predicted Rounded Rank: {rounded_rank}")