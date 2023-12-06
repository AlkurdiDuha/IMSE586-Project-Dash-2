# Importing the necessary libraries
import numpy as np
import pandas as pd
from tabulate import tabulate
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier  # To perform the KNN algorithm
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc  # These functions from sklearn.metrics are used for evaluating the performance of the machine learning model. 
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler  # For scaling the data
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import statsmodels.api as sm
# ------------------------------------------------------------ #
# Reading the .csv file that is labelled as "hepatitis.csv" into Pandas dataframe with the name df
df = pd.read_csv("./Data/hepatitis.csv")

# Exploratory Visuals
# Defining the categorical and numerical columns
cat_cols = ['steroid', 'antivirals', 'fatigue', 'malaise', 'anorexia',
                    'liver_big', 'liver_firm', 'spleen_palpable', 'spiders', 'ascites',
                    'varices', 'histology']  # The 'sex' will be used as hue
num_cols = ['age', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime']


# ------------------------------------------------------------ #

# Handling Missing Data
# Drop rows with missing categorical data
df.dropna(subset=['steroid', 'fatigue', 'malaise', 'anorexia', 'liver_big', 'liver_firm', 'spleen_palpable', 'spiders', 'ascites', 'varices'], inplace=True)

# Count the total number of rows remaining in the dataframe
total_rows = df.shape[0]

# Count the number of missing values in each numerical column
missing_values_in_num_cols = df[num_cols].isna().sum()

print(f"Total rows after removing missing categorical data: {total_rows}")
print("Missing values in each numerical column:")
print(missing_values_in_num_cols)


# Approach (1): Fill missing values in numerical columns with the mean of each column
for col in num_cols:
    df[col].fillna(df[col].mean(), inplace=True)
# ------------------------------------------------------------ #
# Converting categorical columns into numerical columns
# Identify categorical columns that need to be converted to numerical format for machine learning models
cat_cols = ['class', 'sex', 'steroid', 'antivirals', 'fatigue', 'malaise', 'anorexia',
                    'liver_big', 'liver_firm', 'spleen_palpable', 'spiders', 'ascites',
                    'varices', 'histology']

# Loop through each categorical column to convert them into dummy/indicator variables
for col in cat_cols:
    # Convert each categorical column into dummy variables. 
    # where each unique value in the column becomes a new binary column (0 or 1).
    # 'drop_first=True' omits the first category to prevent multicollinearity (dummy variable trap).
    # 'dtype=int' ensures the new columns are of integer type.
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)

    # Concatenate the new dummy variables to the original DataFrame.
    df = pd.concat([df, dummies], axis=1)
# ------------------------------------------------------------ #
# ------------------------------------------------------------ #
# METHOD_1: Using the statsmodels
# Identify Predictor Variables
# Define the numeric columns you want to include
num_cols = ['age', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime']  # Previosuly defined (Added for clarity)

# Dummy columns: selecting columns that end with 'True'
true_cols = [col for col in df.columns if col.endswith('True')]

# Combine the lists and include the columns that represents the sex as a binary variable
predictors = num_cols + true_cols + ['sex_male']

# Create the Regression Formula
# The regression formula is a string that defines the model to be fitted.
# The format is 'dependent_variable ~ independent_variable1 + independent_variable2 + ...'
# Here, 'class_live' is the dependent variable, and all other columns are independent variables.
# The 'join' function concatenates all predictor column names, separated by ' + ', to create the formula.
formula = 'class_live ~ ' + ' + '.join(predictors)

# ------------------------------------------------------------ #
# METHOD_2: Using the scikit-learn

# Preparing the data for logistic regression
X = df[predictors]  # Using the exact same predictors to define the features (X)
y = df['class_live']  # Define the target variable (y)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
# "test_size=0.2" means 20% of the data is reserved for testing

# Instantiating the logistic regression model
clf = LogisticRegression(max_iter=1000)  # Create a logistic regression classifier
# max_iter=1000 increases the number of iterations to ensure convergence (This part was added to avoid a warning message)

# Setting up cross-validation parameters
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=123)  
# RepeatedKFold splits the data into 5 parts (folds), repeating this process 3 times
# random_state=123 ensures that the splits are reproducible and consistent across runs

# Performing the cross-validation on the training data
scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=cv)  
# cross_val_score evaluates the model (clf) using the training data with defined cross-validation (cv)
# It returns a list of accuracy scores, one for each fold of each repeat (3 x 5)

# Displaying the results of cross-validation
print("Cross-Validation Accuracies for each fold and repeat:")
print(scores)  # Prints the accuracy of each fold in each repeat

# Calculating the average performance across all folds and repeats
mean_cv_accuracy = np.mean(scores)  # Computes the mean accuracy of cross-validation
print(f"Mean Cross-Validation Accuracy: {mean_cv_accuracy:.4f}")

# Fitting the model on the entire training set
clf.fit(X_train, y_train)

# Evaluating the model on the test set
test_accuracy = clf.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
# Prints the accuracy on the test set to understand the model's performance on unseen data
# ------------------------------------------------------------ #
pred_num_col = [
    'age', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime'
]



logit_model = sm.Logit(df['class_live'], sm.add_constant(df[pred_num_col]))
result = logit_model.fit()


app = dash.Dash(__name__)

server = app.server

app.layout = html.Div([
    html.H1("Numerical Data - LR"),

    dcc.Dropdown(
        id='predictor-dropdown',
        options=[{'label': col, 'value': col} for col in pred_num_col],
        value='age',
        style={'width': '50%'}
    ),

    dcc.Graph(id='probability-plot'),
])

@app.callback(
    Output('probability-plot', 'figure'),
    [Input('predictor-dropdown', 'value')]
)
def update_plots(selected_predictor):
    prob_live = result.predict(sm.add_constant(df[pred_num_col]))
    df['prob_live'] = prob_live
    prob_die = 1 - prob_live
    df['prob_die'] = prob_die

    df['class_live_color'] = df['class_live'].map({1: 'live', 0: 'die'})
    probability_fig = px.scatter(
        df,
        x=selected_predictor,
        y='prob_live' if selected_predictor != 'sex_male' else 'prob_die',
        color='class_live_color',
        labels={'x': selected_predictor, 'y': 'Probability of Death'},
        title=f'Probability Plot: {selected_predictor} vs. class_live'
    )

    return probability_fig


if __name__ == '__main__':
    app.run_server(jupyter_mode = 'external', debug=True, port=8051)