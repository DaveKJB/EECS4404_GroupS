import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


data = pandas.read_csv("data.csv")

feature_names = ['GPA', 'SAT','TOEFL']
X = data[feature_names]
y = data['Rank']

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using all the data
regr.fit(X, y)

# Display the linear regression equation
equation = f"Rank = {regr.intercept_} + {regr.coef_[0]} * GPA + {regr.coef_[1]} * SAT + {regr.coef_[2]} * TOEFL"
print()
print(f"Linear Regression Equation: {equation}")
print()
print('Close the graph and input new GPA, SAT, and TOEFL for predicted Ranking')

feature_names = ['GPA', 'TOEFL']
regr2 = linear_model.LinearRegression()
regr2.fit(X, y)


feature_names = ['GPA', 'SAT']
regr3 = linear_model.LinearRegression()
regr3.fit(X, y)


# Subsetting the first 100 samples
subset_data = data.head(100)

# First graph: GPA and SAT vs Rank
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')

# Scatter plot
ax1.scatter(subset_data['GPA'], subset_data['SAT'], subset_data['Rank'], c='blue', marker='o')

# Regression plane for regr
xx1, xx2 = np.meshgrid(np.linspace(data['GPA'].min(), data['GPA'].max(), 100),
                       np.linspace(data['SAT'].min(), data['SAT'].max(), 100))
zz1 = regr3.intercept_ + regr3.coef_[0] * xx1 + regr3.coef_[1] * xx2
ax1.plot_surface(xx1, xx2, zz1, alpha=0.5, color='red')

ax1.set_xlabel('GPA')
ax1.set_ylabel('SAT')
ax1.set_zlabel('Rank')
ax1.set_title('GPA and SAT vs Rank')

# Second graph: GPA and TOEFL vs Rank
ax2 = fig.add_subplot(122, projection='3d')

# Scatter plot
ax2.scatter(data['GPA'], data['TOEFL'], data['Rank'], c='red', marker='o')

# Regression plane for regr2
xx3, xx4 = np.meshgrid(np.linspace(data['GPA'].min(), data['GPA'].max(), 100),
                       np.linspace(data['TOEFL'].min(), data['TOEFL'].max(), 100))
zz2 = regr2.intercept_ + regr2.coef_[0] * xx3 + regr2.coef_[2] * xx4
ax2.plot_surface(xx3, xx4, zz2, alpha=0.5, color='blue')

ax2.set_xlabel('GPA')
ax2.set_ylabel('TOEFL')
ax2.set_zlabel('Rank')
ax2.set_title('GPA and TOEFL vs Rank')

plt.show()


def get_valid_input(prompt, min_value, max_value):
    while True:
        try:
            user_input = input(prompt)
            
            # Check if the user wants to quit
            if user_input.lower() == 'q':
                return None
            
            # Convert the input to a float
            user_value = float(user_input)
            
            # Check if the input is within the specified range
            if min_value <= user_value <= max_value:
                return user_value
            else:
                print(f"Please enter a value between {min_value} and {max_value}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Take user input for GPA
user_gpa = get_valid_input("Enter GPA (0-4) or 'q' to quit: ", 0, 4)
if user_gpa is None:
    exit()

# Take user input for SAT
user_sat = get_valid_input("Enter SAT score (400-1600) or 'q' to quit: ", 400, 1600)
if user_sat is None:
    exit()

# Take user input for TOEFL
user_toefl = get_valid_input("Enter TOEFL score (0-120) or 'q' to quit: ", 0, 120)
if user_toefl is None:
    exit()

# Continue with the rest of your program using user_gpa, user_sat, and user_toefl


# Make a prediction using the trained model
user_data = np.array([[user_gpa, user_sat, user_toefl]])
predicted_rank = regr.predict(user_data)

# Round the predicted rank to the nearest positive whole number
rounded_rank = round(predicted_rank[0])

# Display the rounded predicted rank
print()
print(f"Predicted Rounded Rank: {rounded_rank}")