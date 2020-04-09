# Prediction Model - House Prices

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Display all the columns of the dataframe
pd.pandas.set_option('display.max_columns', None)

dataset = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
y = dataset.iloc[:, 80].values

dataset.shape

dataset.head()

# Checking % of NaN values present in each feature

## List of features which has missing values
features_with_nan= [features for features in dataset.columns if dataset[features].isnull().sum()>1]

## Printing % of missing values for every feature
for feature in features_with_nan:
    print (feature, np.round(dataset[feature].isnull().mean(), 4), '% missing values')
    
# Finding relationship between missing values and sales price
for feature in features_with_nan:
   data = dataset.copy()
    
   # Transforming missing values to 1 and anything else to 0
   data[feature] = np.where(data[feature].isnull(), 1, 0)
   
   # Correlating the mean SalePrice where the information is missing (1) or present (1)
   data.groupby(feature)['SalePrice'].median().plot.bar()
   plt.title(feature)
   plt.show()

for feature in features_with_nan:
   data_test = df_test.copy()
    
   # Transforming missing values to 1 and anything else to 0
   data_test[feature] = np.where(df_test[feature].isnull(), 1, 0)
   
### There is a relation between missing values and the dependent variable (SalePrice). It is necessary to handle 
### the NaN values

# Numerical categories
numerical_features = [features for features in dataset.columns if dataset[features].dtypes != 'O']
print ('Number of numerical variables: ', len(numerical_features))

# Visualising the numerical variables
dataset[numerical_features].head()

# Temporal categories (Categories with years)
year_feature = [features for features in numerical_features if 'Yr' in features or 'Year' in features]

# Displaying year list
year_feature

# Checking content of these year variables
for feature in year_feature:
    print (feature, dataset[feature].unique())
    
# Correlating the YrSold with SalePrice
data.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title('House Price vs YearSold')
plt.show()

# Comparing other year_feature with SalePrice
for feature in year_feature:
    if feature != 'YrSold': 
        data=dataset.copy()
        
        data[feature] = data['YrSold'] - data[feature]
        
        plt.scatter(data[feature], data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()
        
# Numerical variables: Continuous and Discrete
        
## Discrete
discrete_features = [feature for feature in numerical_features if len(dataset[feature].unique()) < 25 and feature not in year_feature + ['Id']]
print ('Number of discrete variables: ', len(discrete_features))

discrete_features

dataset[discrete_features].head()

# Correlating discrete_features with SalePrice
for feature in discrete_features:
    data = dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()

## Continuous
continous_features = [feature for feature in numerical_features if feature not in discrete_features + year_feature + ['Id']]
print ('Continous features: {}'.format(len(continous_features)))

# Correlating continous_features with SalePrice (recomend use of histograms to understand)
for feature in continous_features:
    data = dataset.copy()
    data[feature].hist(bins = 25)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.title(feature)
    plt.show()

# Transforming data using log transformationg
data = dataset.copy()

for feature in continous_features:
    data = dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data['SalePrice'] = np.log(data['SalePrice'])
        plt.scatter(data[feature], data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.title(feature)
        plt.show()
        
# Checking outliers (very high or very low values) / Only in continous features
for feature in continous_features:
    data = dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data.boxplot(column = feature)
        plt.xlabel(feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()
        
# Categorical variables
categorical_features = [feature for feature in dataset.columns if data[feature].dtypes == 'O']
categorical_features

dataset[categorical_features].head()

for feature in categorical_features:
    print ('The feature is {} and number of categories are {}'.format(feature, len(dataset[feature].unique())))

# Correlating categorical_features with SalePrice
for feature in categorical_features:
    data = dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()

# Capturing NaN values in categorical_features
features_nan = [feature for feature in dataset.columns if dataset[feature].isnull().sum() > 1 and dataset[feature].dtype=='O']
for feature in features_nan:
    print ('{}: {}% missing values'.format(feature, np.round(dataset[feature].isnull().mean(), 4)))
    
## Replacing missing values with a new label, swapping NaN for Missing
def replace_cat_feature(dataset,features_nan):
    data = dataset.copy()
    data[features_nan] = data[features_nan].fillna('Missing')
    return data

dataset = replace_cat_feature(dataset, features_nan)
df_test = replace_cat_feature(df_test, features_nan)

dataset[features_nan].isnull().sum()
df_test[features_nan].isnull().sum()

dataset.head()

# Capturing NaN values in numerical_features
numerical_nan = [feature for feature in dataset.columns if dataset[feature].isnull().sum() > 1 and dataset[feature].dtype !='O']

for feature in numerical_nan:
    print ('{}: {}% missing values'.format(feature, np.round(dataset[feature].isnull().mean(), 4)))
    
## Replacing NaN of Numerical Values with mean()
for feature in numerical_nan:
    ## Replaceing using median since there are lot of outliers
    median_value = dataset[feature].median()
    
    ## Creating a new feature to capture NaN values
    dataset[feature + 'nan'] = np.where(dataset[feature].isnull(), 1, 0)
    dataset[feature].fillna(median_value, inplace=True)

## Test - Replacing NaN of Numerical Values with mean()    
for feature in numerical_nan:
    ## Replaceing using median since there are lot of outliers
    median_value = df_test[feature].median()
    
    ## Creating a new feature to capture NaN values
    df_test[feature + 'nan'] = np.where(df_test[feature].isnull(), 1, 0)
    df_test[feature].fillna(median_value, inplace=True)
    
dataset[numerical_nan].isnull().sum()
df_test[numerical_nan].isnull().sum()

## Converting temporal varibles to "how many years"

for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    dataset[feature] = dataset['YrSold']-dataset[feature]

# Test dataframe
for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    df_test[feature] = df_test['YrSold']-df_test[feature]
    
dataset.head()
df_test.head()

num_features = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
num_features_test = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']

for feature in num_features:
    dataset[feature] = np.log(dataset[feature])

# Test dataframe
for feature in num_features_test:
    df_test[feature] = np.log(df_test[feature])
    
dataset.head()
df_test.head()

# Handling rare category in features
for feature in categorical_features:
    temp = dataset.groupby(feature)['SalePrice'].count()/len(dataset)
    temp_df = temp[temp > 0.01].index
    dataset[feature] = np.where(dataset[feature].isin(temp_df), dataset[feature], 'Rare_var')

# Test - Handling rare category in features    
for feature in categorical_features:
    temp = df_test.groupby(feature).count()/len(dataset)
    temp_df = temp[temp > 0.01].index
    df_test[feature] = np.where(df_test[feature].isin(temp_df), df_test[feature], 'Rare_var')

dataset.head()
df_test.head()

dataset[categorical_features]
df_test[categorical_features]

# Changing categorical features to values
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for feature in categorical_features:
    dataset[feature] = labelencoder.fit_transform(dataset[feature])
    
for feature in categorical_features:
    df_test[feature] = labelencoder.fit_transform(df_test[feature])
   
cols_drop = ['SalePrice']
dataset.drop(cols_drop, axis = 1, inplace=True)
dataset = np.nan_to_num(dataset)

df_test = np.nan_to_num(df_test)

# Fitting SVR to the dataset
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(dataset, y)

y_pred = regressor.predict(df_test)

submission = pd.DataFrame({
        "Id": data_test["Id"],
        'SalePrice': y_pred
        })

submission.to_csv('submission_housepricever4.csv', index = False)



    