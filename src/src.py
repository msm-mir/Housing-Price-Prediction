import pandas as pd
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# fill null cells
def fillna(df, null_cnt):
    for col_name in null_cnt.index:
        if null_cnt[col_name] > 0:
            median_val = df[col_name].median()
            df[col_name] = df[col_name].fillna(median_val)

# calculate number of outliers and clip them
def clip_outliers(col):
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - (IQR * 1.5)
    upper_bound = Q3 + (IQR * 1.5)

    outliers = col[(col < lower_bound) | (col > upper_bound)]
    clip_col = col.clip(lower_bound, upper_bound)
    return len(outliers), clip_col

# boxplot visualization for demonstrating outliers
def boxplot(col, col_name, time):
    plt.figure(figsize=(7, 4))
    sns.boxplot(y=col)
    plt.title(f'Outliers of {col_name} {time} clip')
    plt.show()

# scatterplot visualization for demonstrating correlation between each feature and target
def scatterplot_vs_Price(col_name):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x=col_name, y='Price')
    plt.title(f'Scatterplot: {col_name} vs Price')
    plt.show()

# regplot visualization for demonstrating correlation between each feature and target and regression line
def regplot_vs_Price(df, col_name, o):
    plt.figure(figsize=(8, 5))
    sns.regplot(data=df, x=col_name, y='Price', order=o)
    plt.title(f'Regplot: {col_name} vs Price')
    plt.show()

# read dataset and split it into training and testing sets
df = pd.read_csv('house-prices.csv')
X = df[['SqFt', 'Bedrooms', 'Bathrooms', 'Offers', 'Brick', 'Neighborhood']].copy()
y = df['Price'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# output number of train and test data
print(f'Number of data for train: {len(X_train)}')
print(f'Number of data for test: {len(X_test)}')
print(f'Number of all data: {len(df)} \n')

# output information of dataset
print('Info function:')
print(f'{df.info()}\n')

print(f'Isna function:\n{df.isna().sum()}\n')

# fill null cells with column median 
null_cnt = df.isna().sum()
fillna(df, null_cnt)

null_cnt = X_train.isna().sum()
fillna(X_train, null_cnt)

# output number of null cells
total_nulls = df.isna().sum().sum()
if total_nulls == 0:
    print('There is no null values left in data frame! \n')
else:
    print(f'There are still {total_nulls} null values left in data frame! \n')

# boxplot before clip outliers
boxplot(y_train, 'Price', 'Before')
boxplot(X_train['SqFt'], 'SqFt', 'Before')

print(f'Describe function:\n{df.describe()}\n')

# check for outliers
# Price outliers and clipping them
Price_outliers_num, y_train = clip_outliers(y_train)
print(f'Number of outliers of Price: {Price_outliers_num}')

# SqFt outliers and clipping them
SqFt_outliers_num, X_train['SqFt'] = clip_outliers(X_train['SqFt'])
print(f'Number of outliers of SqFt: {SqFt_outliers_num} \n')

# output for demonstrate outliers have been clipped
Price_outliers_num, _ = clip_outliers(y_train)
print(f'Number of outliers of Price (After clip columns): {Price_outliers_num}')

SqFt_outliers_num, _ = clip_outliers(X_train['SqFt'])
print(f'Number of outliers of SqFt (After clip columns): {SqFt_outliers_num} \n')

# boxplot after clip outliers
boxplot(y_train, 'Price', 'After')
boxplot(X_train['SqFt'], 'SqFt', 'After')

# dispersion of discrete data
print(f'Dispersion of {df['Bedrooms'].value_counts(normalize=True) * 100} \n')
print(f'Dispersion of {df['Bathrooms'].value_counts(normalize=True) * 100} \n')
print(f'Dispersion of {df['Offers'].value_counts(normalize=True) * 100} \n')

# visualisation
# convert Brick values to binary values
df['Brick'] = df['Brick'].map({'No': 0, 'Yes': 1})
X_train['Brick'] = X_train['Brick'].map({'No': 0, 'Yes': 1})
X_test['Brick'] = X_test['Brick'].map({'No': 0, 'Yes': 1})

# convert Neighborhood values to numeric values
df['Neighborhood'] = df['Neighborhood'].map({'North': 0, 'East': 1, 'South':2, 'West': 3})

# split Neighborhood to three column with binary values
one_hot = pd.get_dummies(X_train['Neighborhood'], prefix='Neighborhood')
X_train = pd.concat([X_train, one_hot], axis=1)
X_train = X_train.drop('Neighborhood', axis=1)

one_hot = pd.get_dummies(X_test['Neighborhood'], prefix='Neighborhood')
X_test = pd.concat([X_test, one_hot], axis=1)
X_test = X_test.drop('Neighborhood', axis=1)

# scatterplot for Price and other columns
scatterplot_vs_Price('SqFt')
scatterplot_vs_Price('Bedrooms')
scatterplot_vs_Price('Bathrooms')
scatterplot_vs_Price('Offers')
scatterplot_vs_Price('Brick')
scatterplot_vs_Price('Neighborhood')

# correlation matrix by heatmap
plt.figure(figsize=(9, 5))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmap(Correlations): Price vs Features")
plt.show()

# regplot for Price and other columns
regplot_vs_Price(df, 'SqFt', 1)
regplot_vs_Price(df, 'Bedrooms', 1)
regplot_vs_Price(df, 'Bathrooms', 1)
regplot_vs_Price(df, 'Offers', 1)
regplot_vs_Price(df, 'Brick', 1)
regplot_vs_Price(df, 'Neighborhood', 1)

# model implementation
# normalize dataset by standardize
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

std = np.where(std == 0, 1e-10, std)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# convert dataframe to numpy array
X_train = X_train.values.astype(float)
X_test = X_test.values.astype(float)

y_train = y_train.values.reshape(-1, 1).astype(float)
y_test = y_test.values.reshape(-1, 1).astype(float)

# add a column with ones to the beginning of X
ones_col = np.ones((X_train.shape[0], 1))
X_train = np.column_stack((ones_col, X_train))

ones_col = np.ones((X_test.shape[0], 1))
X_test = np.column_stack((ones_col, X_test))

# model class
class GDLR:
    def __init__(self, learning_rate, epoch):
        self.parameter = None
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.all_rmse = []

    # prediction
    def prediction(self, X):
        return np.dot(X, self.parameter)
    
    # cost function by RMSE
    def rmse_comp(self, predict, y):
        err = predict - y
        # if (y.shape[0] < 20): print(abs(err))
        mse = np.mean(err ** 2)
        rmse = np.sqrt(mse)
        return rmse
    
    # fit function
    def fit(self, X, y):
        # parameters (weights) initialization
        self.parameter = np.zeros((X.shape[1], 1))

        for _ in range(self.epoch):
            predict = self.prediction(X)
            rmse = self.rmse_comp(predict, y)
            self.all_rmse.append(rmse)
            gradient = (np.dot(X.T, (predict - y))) / (y.shape[0] * (rmse + 1e-10))
            self.parameter -= (self.learning_rate * gradient)

# create model instance
my_model = GDLR(learning_rate=0.8, epoch=260_000)
my_model.fit(X_train, y_train)

# output of cost function for training and test data
print(f'RMSE for training data: {my_model.rmse_comp(my_model.prediction(X_train), y_train):.2f}')
print(f'RMSE for testing data: {my_model.rmse_comp(my_model.prediction(X_test), y_test):.2f}\n\n\n\n')

# plotting the convergence curve
plt.figure(figsize=(8, 5))
plt.plot(range(len(my_model.all_rmse)), my_model.all_rmse)
plt.xlabel('epoch')
plt.ylabel('RMSE')
plt.title('Training RMSE')
plt.grid(True)
plt.show()