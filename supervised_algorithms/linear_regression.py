import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the  Housing dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["target"] = housing.target

# Select the features with correlation coefficient greater than 0.5
corr_matrix = df.corr()
print(sns.heatmap(corr_matrix, cmap="coolwarm", annot=True))

# Define the predictor and target variables
X = df[["MedInc", "AveBedrms", "Latitude", "Longitude"]]
y = df.target

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the training features using the StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

# Add a constant term to the independent variables (required for the regression)
X_train_scaled = sm.add_constant(X_train_scaled)
X_test_scaled = sm.add_constant(X_test_scaled)

# Create the linear regression model on the training set
model = sm.OLS(y_train, X_train_scaled).fit()

# Output the table with p-values and adjusted R-squared for the test set
print(model.summary(yname='y_test', xname=pd.DataFrame(X_train_scaled).columns.tolist()))
