from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load the data into a pandas DataFrame
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['target'] = housing.target

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)

# Create a PCA object with the desired number of components
pca = PCA(n_components=5)

# Fit the PCA model to the standardized data
pca.fit(data_scaled)

# Transform the data into the new set of principal components
data_transformed = pca.transform(data_scaled)

# Print the variance explained by each component
print(pca.explained_variance_ratio_)
