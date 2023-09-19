# %% [markdown]
# # Car Selling Price Prediction


# %% [markdown]
# 
# The dataset contains various attributes of cars, which will be used to predict their prices. There are total of 13 given columns as follows:
# 
#         1. name             : name of the car model
#         2. year             : built year
#         3. selling_price    : current selling price of the car
#         4. km_driven        : total distance driven (in km)
#         5. fuel             : fuel type used
#         6. seller_type      : type of seller
#         7. transmission     : transmission type (manual or automatic)
#         8. owner            : the ownerership order (ith-hand)
#         9. mileage          : the distance it can travel using a unit of fuel
#         10. engine          : the size of engine
#         11. max_power       : car's maximum power (rate of completing work in a timeframe)
#         12. torque          : car's torque (capacity to do work)
#         13. seats           : the number of seats avaialable
# 
# In this notebook, a selling_price prediction model will be created using other atrributes as features.

# %% [markdown]
# 
# ### Importing Libraries

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Importing Data

# %%
df = pd.read_csv('code/code/data/Cars - Cars.csv')

# %%
# visualize the head and tail of dataframe
df

# %%
# dropping torque column
df.drop(['torque'], axis=1, inplace=True)

# %%
df.shape

# %%
df.info()

# %%
df.describe()

# %% [markdown]
# Because we are predicting *selling_price* which is a continuous label, we will try to implement a regressional model but there are only 4 applicable features so far.
# We will inspect more on those string-type values to see the possiblity to encode them.

# %%
df.iloc[:,4:8].head()

# %%
# counting the number of unique values of each columns above.

for col in df.iloc[:,4:8].columns:
    a = len(df[col].unique())
    print(f'The number of unique values of {col}: {a}')
    if a < 10:
        print(f'    Unique values of {col}: {df[col].unique()}')


# %%
df['fuel'].value_counts()

# %% [markdown]
# *mileage*, *engine*, *max_power*, and *torque* can be converted to number if their units are the same and, therfore, neglected.
# 
# The *mileage* of natural gas-powered cars use different system from petrol-powered cars, so we will exclude natural gas-powered cars from the dataset. In addition, natural gas-powered cars have very few samples (95 cars).

# %%
df.drop(df[df['fuel'] == 'CNG'].index, inplace=True)
df.drop(df[df['fuel'] == 'LPG'].index, inplace=True)

# %%
# observing units in each columns
# considering if we have to convert some units or not

print(f'Unique ending of "mileage": {df["mileage"].str[-5:].unique()}')
print(f'Unique ending of "engine": {df["engine"].str[-3:].unique()}')
print(f'Unique ending of "max_power": {df["max_power"].str[-4:].unique()}')


# %%
# no unit conversion is needed
# stripping the unit and change those columns to proper numeric format

df['mileage'] = df['mileage'].str.strip(' kmpl')
df['engine'] = df['engine'].str.strip(' CC')
df['max_power'] = df['max_power'].str.strip(' bhp')

df['mileage'] = df['mileage'].astype(float)
df['engine'] = df['engine'].astype(float)
df['max_power'] = df['max_power'].astype(float)

# %%
# Extract brand from car's name and replace name column

df['name'] = list(df['name'].str.split(' ', n=1).str.get(0))
df.rename(columns={'name':'brand'}, inplace=True)

# %%
df.head()

# %% [markdown]
# ### Feature Encoding

# %%
mp = dict()
owner_mapping = {'Test Drive Car': 5,
                 'First Owner': 1,
                 'Second Owner': 2,
                 'Third Owner': 3,
                 'Fourth & Above Owner': 4}
mp['owner_mapping'] = owner_mapping
df.replace({'owner': owner_mapping}, inplace=True)

# %%
df.head()

# %%
df.info()

# %%
# Import another library
from sklearn.preprocessing import LabelEncoder

# Encode 'fuel' and 'transmission' and print the encoding matrices

le_fuel = LabelEncoder()
le_fuel.fit(df['fuel'])
encoded, mapping = pd.Series(list(le_fuel.classes_)).factorize()
mp['fuel_mapping'] = mapping
print(pd.DataFrame({'': mapping, 'encoded value': encoded}).set_index(''))
df['fuel'] = le_fuel.transform(df['fuel'])

le_transmission = LabelEncoder()
le_transmission.fit(df['transmission'])
encoded, mapping = pd.Series(list(le_transmission.classes_)).factorize()
mp['transmission_mapping'] = mapping
print(pd.DataFrame({'': mapping, 'encoded value': encoded}).set_index(''))
df['transmission'] = le_transmission.transform(df['transmission'])

# %%
df = pd.get_dummies(df, columns=['seller_type'], drop_first=True, dtype=int)

# %% [markdown]
# *seller_type* has 2 dummies columns. 
# - 1 in *seller_type_Individual* -> Individual
# - 1 in *seller_type_Trusmark Dealer* -> Trustmark Dealer
# - 0 in **both** columns -> Dealer

# %%
df.head()

# %% [markdown]
# ## Exploratory Data Analysis

# %%
# Observe 'owner' effect on selling_price
sns.boxplot(data=df, x='owner', y='selling_price')

# %%
# The test drive car selling_price is extremely expensive. We will exclude them.

df = df[df['owner'] != 5]

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# plotting boxplot for car selling price by brand

plt.figure(figsize=(10, 6))
sns.boxplot(x='brand', y='selling_price', data=df)
plt.title('Car Price Distribution by Brand')
plt.xlabel('Brand')
plt.ylabel('Price')
plt.xticks(rotation=90)
plt.show()

# %% [markdown]
# Cars from each brand has different selling price range. The brand origin (continent of origin) shows that European cars generally have higher price than Asian and American cars (inspected leter in Feature Engineering section). But, the third boxplot shows that, among cars from the same origin, there is still some difference in pricing between brand.

# %%
df.info()

# %%
cols = ['year', 'selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 
        'owner', 'fuel', 'transmission', 'seats']
fig, axi = plt.subplots(2, 5, figsize=(25,8))

for i,col in enumerate(cols):
    if i<6:
        sns.histplot(data=df, x=col, bins=20, ax=axi[i//5, i%5])
    else:
        sns.countplot(data=df, x=col, ax=axi[i//5, i%5])
          
plt.show()

# %%
cols = ['owner', 'fuel', 'transmission', 'seats']
fig, axi = plt.subplots(2, 2, figsize=(12,8))

for i,col in enumerate(cols):
    sns.boxplot(data=df, x=col, y='selling_price', ax=axi[i//2, i%2])
          
plt.show()

# %%
cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power']
fig, axi = plt.subplots(2, 3, figsize=(18,8))

for i,col in enumerate(cols):
    sns.scatterplot(data=df, x=col, y='selling_price', ax=axi[i//3,i%3])
sns.scatterplot(data=df, x='engine', y='max_power', ax=axi[1, 2], color='green')

plt.show()

# %% [markdown]
# It seems that 'mileage' will be reaaly bad for 'selling_price' prediction
# 
# On the other hand, 'engine' and 'max_power' seems to be correlated. They might be able to predict and fill each other values if it is unspecified or None. (spoiler alert: it provides no significant improvement on the model performance)

# %% [markdown]
# ### Correlation Matrix

# %%
plt.figure(figsize = (15,8))
sns.heatmap(df.select_dtypes(exclude='object').corr(), annot=True, cmap="coolwarm")

# %% [markdown]
# ### Predictive Power Score Matrix

# %%
import ppscore as pps


matrix_df = pps.matrix(df.select_dtypes(exclude='object'))[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
plt.figure(figsize = (15,8))
sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)

# %% [markdown]
# ## Feature Selection

# %%
df.head()

# %%
# From correlation and ppscore map, we will drop several columns that potentially complicate the model rather than improving performances.

df.drop(['fuel','owner','mileage', 'km_driven', 'seller_type_Individual', 'seller_type_Trustmark Dealer', 'seats'], axis=1, inplace=True)

# %%
X = df.drop(['selling_price'], axis=1)
y = df['selling_price']

# %%
X.info()

# %% [markdown]
# ### Train Test Split

# %% [markdown]
# ## Preprocessing

# %%
# Check Null values

X.isna().sum()

# %%
y.isna().sum()

# no null in selling_price label

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Apply logarithmic scale on y to improve model stabiliy
y_train = np.log(y_train)

# %% [markdown]
# ### Fill missing value

# %%
X

# %%
mp

# %%
# Fill missing 'engine' value

filling = dict()
filling['engine'] = X_train['engine'].median() 

# %%
X_train.isna().sum()

# %%
X_train.shape

# %%
filling

# %%
X_train.info()

# %%
# Create other predictors filling matrices

for col in ['year', 'max_power']:
    filling[col] = X_train[col].median() 
for col in ['transmission']:
    filling[col] = X_train[col].mode()[0]

# %%
filling

# %%
# apply filling to X_train
for col in list(filling.keys()):
    X_train[col].fillna(filling[col], inplace=True)

# %%
# apply filling to X_test
for col in list(filling.keys()):
    X_test[col].fillna(filling[col], inplace=True)

# %%
# all NaN is filled
X_train

# %%
X_train.isna().sum()

# %% [markdown]
# ## Check Outliers

# %%
def outlier_count(col, data = X_train):
    
    q75, q25 = np.percentile(data[col], [75, 25])
    iqr = q75 - q25
    
    min_val = q25 - (iqr*1.5)
    max_val = q75 + (iqr*1.5)
    
    outlier_count = len(np.where((data[col] > max_val) | (data[col] < min_val))[0])
    
    outlier_percent = round(outlier_count/len(data[col])*100, 2)
    
    if(outlier_count > 0):
        print("\n"+15*'-' + col + 15*'-'+"\n")
        print('Number of outliers: {}'.format(outlier_count))
        print('Percent of data that is outlier: {}%'.format(outlier_percent))

# %%
for col in X_train.select_dtypes(exclude='object').columns:
    outlier_count(col)

# %%
X_train.info()

# %% [markdown]
# ## Feature Engineering

# %% [markdown]
# Create polynomial form of *engine* and *max_power* of the order of 2 and 3 for polynomial regression model

# %%
X_train_poly = X_train.copy(deep=True)
X_test_poly = X_test.copy(deep=True)

# %%
X_train_poly['engine2'] = X_train_poly['engine'] ** 2
X_train_poly['max_power2'] = X_train_poly['max_power'] ** 2
X_train_poly['engine3'] = X_train_poly['engine'] ** 3
X_train_poly['max_power3'] = X_train_poly['max_power'] ** 3

# %% [markdown]
# ## Scaling

# %%
# The four features to be scaled
X_train[['year','engine', 'max_power']]

# %%
from sklearn.preprocessing import StandardScaler

# feature scaling helps improve reach convergence faster
to_be_scaled_cols = ['year', 'engine', 'max_power']

scaler = StandardScaler()
X_train[to_be_scaled_cols] = scaler.fit_transform(X_train[to_be_scaled_cols])

# %%
# save the scaler for later uses

import pickle
pickle.dump(scaler, open('code/code/model/scaler.pkl','wb'))

# %%
X_train

# %%
# one-hot encode the brand column
brand_list = list(X['brand'].values)

brand_dm = pd.get_dummies(X['brand'], columns=['brand'], dtype=int, prefix='b')
brand_dm = brand_dm.columns

col_order = ['year','transmission','engine','max_power']
col_order.extend(sorted(brand_dm))

X_train = pd.get_dummies(X_train, columns=['brand'], dtype=int, prefix='b')

missing_cols = set(brand_dm) - set(X_train.columns)
for col in missing_cols:
    X_train[col] = 0

X_train = X_train[col_order]

# %%
col_order

# %%
def fill_scale_encode(df, brand_dm, scaler, col_order, to_be_scaled_cols):                     # For filling NaN and scale value in features df (X)

    df[to_be_scaled_cols]  = scaler.transform(df[to_be_scaled_cols])

    df = pd.get_dummies(df, columns=['brand'], dtype=int, prefix='b')
    missing_cols = set(brand_dm) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    
    return df[col_order]

# %%
X_test = fill_scale_encode(X_test, brand_dm, scaler, col_order, to_be_scaled_cols)

# %%
# Check shape and NaN values in test and train set

print(f"Shape of X_train: {X_train.shape} —————————— null value: {X_train.isna().sum().sum()}")
print(f"Shape of X_test: {X_test.shape} —————————— null value: {X_test.isna().sum().sum()}")
print(f"Shape of y_train: {y_train.shape} —————————— null value: {y_train.isna().sum().sum()}")
print(f"Shape of y_test: {y_test.shape} —————————— null value: {y_test.isna().sum().sum()}")

# %% [markdown]
# Scale the polynomial dataset

# %%
from sklearn.preprocessing import StandardScaler

# feature scaling helps improve reach convergence faster
to_be_scaled_cols_poly = ['year', 'engine', 'max_power', 'engine2', 'max_power2','engine3', 'max_power3']

scaler_poly = StandardScaler()
X_train_poly[to_be_scaled_cols_poly] = scaler_poly.fit_transform(X_train_poly[to_be_scaled_cols_poly])
pickle.dump(scaler_poly, open('code/code/model/scaler_poly.pkl','wb'))

# one-hot encode the brand column
brand_list = list(X['brand'].values)

brand_dm = pd.get_dummies(X['brand'], columns=['brand'], dtype=int, prefix='b')
brand_dm = brand_dm.columns

col_order_poly = ['year','transmission','engine','max_power','engine2','max_power2','engine3','max_power3']
col_order_poly.extend(sorted(brand_dm))

X_train_poly = pd.get_dummies(X_train_poly, columns=['brand'], dtype=int, prefix='b')

missing_cols = set(brand_dm) - set(X_train_poly.columns)
for col in missing_cols:
    X_train_poly[col] = 0

X_train_poly = X_train_poly[col_order_poly]

# %%
X_test_poly['engine2'] = X_test_poly['engine'] ** 2
X_test_poly['max_power2'] = X_test_poly['max_power'] ** 2
X_test_poly['engine3'] = X_test_poly['engine'] ** 3
X_test_poly['max_power3'] = X_test_poly['max_power'] ** 3

X_test_poly = fill_scale_encode(X_test_poly, brand_dm, scaler_poly, col_order_poly, to_be_scaled_cols_poly)

# %% [markdown]
# ## Modeling

# %% [markdown]
# ### Redefine LinearRegression Model

# %%
import mlflow
import os

mlflow.set_tracking_uri("http://localhost:5000")
os.environ["LOGNAME"] = "ksung"
print('Creating experiment')
# mlflow.create_experiment("ksung-mod-linreg2")
mlflow.set_experiment("ksung-mod-linreg2")

# %%
intercept = np.ones((X_train.shape[0], 1))
X_train   = np.concatenate((intercept, X_train), axis=1)
X_train_poly   = np.concatenate((intercept, X_train_poly), axis=1)
intercept = np.ones((X_test.shape[0], 1))
X_test    = np.concatenate((intercept, X_test), axis=1)
X_test_poly    = np.concatenate((intercept, X_test_poly), axis=1)

# %%
X_train_lin = np.copy(X_train)
X_test_lin = np.copy(X_test)
X_train = None
X_test = None

# %%
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# %%
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

class LinearRegression(object):
    
    #in this class, we add cross validation as well for some spicy code....
    kfold = KFold(n_splits=3)
            
    def __init__(self, regularization, poly, momentum, method, initial_method, lr, cv=kfold, num_epochs=15000, batch_size=50):
        self.lr         = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method     = method
        self.cv         = cv
        self.regularization = regularization
        self.initial_method = initial_method
        self.momentum = momentum
        self.previous_step = 0
        self.poly = poly

    def mse(self, ytrue, ypred):
        return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]
        # return ((np.exp(ypred) - np.exp(ytrue)) ** 2).sum() / ytrue.shape[0]
    
    def r2_score(self, ytrue, ypred):
        ymean = sum(ytrue) / len(ytrue)
        return 1 - (((ypred - ytrue) ** 2).sum() / ((ymean - ytrue) ** 2).sum())
    
    def fit(self, X_train, y_train):
            
        #create a list of kfold scores
        self.kfold_scores = list()
        
        #reset val loss
        self.val_loss_old = np.infty

        #kfold.split in the sklearn.....
        #5 splits
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val   = X_train[val_idx]
            y_cross_val   = y_train[val_idx]
            
            if self.initial_method == 'zero':
                self.theta = np.zeros(X_cross_train.shape[1])
            elif self.initial_method == 'xavier':
                m = X_cross_train.shape[1]
                lower, upper = -(1.0 / np.sqrt(m)), (1.0 / np.sqrt(m))
                self.theta = np.random.uniform(lower, upper, size=X_cross_train.shape[1])
            
            #define X_cross_train as only a subset of the data
            #how big is this subset?  => mini-batch size ==> 50
            
            #one epoch will exhaust the WHOLE training set
            with mlflow.start_run(run_name=f"Fold-{fold}", nested=True):
                
                params = {"method": self.method, "lr": self.lr, "reg": type(self).__name__, 
                            "poly": bool(self.poly), "momentum": self.momentum, 
                            "initial_method": self.initial_method}
                mlflow.log_params(params=params)
                for epoch in range(self.num_epochs):
                
                    #with replacement or no replacement
                    #with replacement means just randomize
                    #with no replacement means 0:50, 51:100, 101:150, ......300:323
                    #shuffle the index
                    perm = np.random.permutation(X_cross_train.shape[0])
                            
                    X_cross_train = X_cross_train[perm]
                    y_cross_train = y_cross_train[perm]
                    
                    if self.method == 'sto':
                        for batch_idx in range(X_cross_train.shape[0]):
                            X_method_train = X_cross_train[batch_idx].reshape(1, -1) #(11,) ==> (1, 11) ==> (m, n)
                            y_method_train = np.array([y_cross_train[batch_idx]])
                            train_loss = self._train(X_method_train, y_method_train)
                    elif self.method == 'mini':
                        for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                            #batch_idx = 0, 50, 100, 150
                            X_method_train = X_cross_train[batch_idx:batch_idx+self.batch_size, :]
                            y_method_train = y_cross_train[batch_idx:batch_idx+self.batch_size]
                            train_loss = self._train(X_method_train, y_method_train)
                    else:
                        X_method_train = X_cross_train
                        y_method_train = y_cross_train
                        train_loss = self._train(X_method_train, y_method_train)

                    mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)

                    yhat_val = self.predict(X_cross_val)
                    val_loss_new = self.mse(y_cross_val, yhat_val)
                    mlflow.log_metric(key="val_loss", value=val_loss_new, step=epoch)
                    
                    #record dataset
                    mlflow_train_data = mlflow.data.from_numpy(features=X_method_train, targets=y_method_train)
                    mlflow.log_input(mlflow_train_data, context="training")
                    
                    mlflow_val_data = mlflow.data.from_numpy(features=X_cross_val, targets=y_cross_val)
                    mlflow.log_input(mlflow_val_data, context="validation")
                    
                    #early stopping
                    if np.allclose(val_loss_new, self.val_loss_old):
                        break
                    self.val_loss_old = val_loss_new
                
                self.kfold_scores.append(val_loss_new)
                print(f"Fold {fold}: {val_loss_new}")      
                    
    def _train(self, X, y):
        yhat = self.predict(X)
        m    = X.shape[0]        
        grad = (1/m) * X.T @(yhat - y) + self.regularization.derivation(self.theta)
        step = self.lr * grad
        self.theta = self.theta - step - self.momentum * self.previous_step
        self.previous_step = step
        return self.mse(y, yhat)
    
    def predict(self, X):
        return X @ self.theta  #===>(m, n) @ (n, )
    
    def _coef(self):
        return self.theta[1:]  #remind that theta is (w0, w1, w2, w3, w4.....wn)
                               #w0 is the bias or the intercept
                               #w1....wn are the weights / coefficients / theta
    def _bias(self):
        return self.theta[0]
    
    def _feature_importance(self, poly=False):
        if not poly:
            col_names = np.array(['year','transmission','engine','max_power','b_Ambassador','b_Ashok','b_Audi',
                    'b_BMW','b_Chevrolet','b_Daewoo','b_Datsun','b_Fiat','b_Force','b_Ford',
                    'b_Honda','b_Hyundai','b_Isuzu','b_Jaguar','b_Jeep','b_Kia','b_Land',
                    'b_Lexus','b_MG','b_Mahindra','b_Maruti','b_Mercedes-Benz','b_Mitsubishi',
                    'b_Nissan','b_Opel','b_Peugeot','b_Renault','b_Skoda','b_Tata',
                    'b_Toyota','b_Volkswagen','b_Volvo'])
        else:
            col_names = np.array(['year','transmission','engine','max_power','engine2','max_power2',
                    'engine3','max_power3','b_Ambassador','b_Ashok','b_Audi',
                    'b_BMW','b_Chevrolet','b_Daewoo','b_Datsun','b_Fiat','b_Force','b_Ford',
                    'b_Honda','b_Hyundai','b_Isuzu','b_Jaguar','b_Jeep','b_Kia','b_Land',
                    'b_Lexus','b_MG','b_Mahindra','b_Maruti','b_Mercedes-Benz','b_Mitsubishi',
                    'b_Nissan','b_Opel','b_Peugeot','b_Renault','b_Skoda','b_Tata',
                    'b_Toyota','b_Volkswagen','b_Volvo'])
        plt.xticks(rotation=90)
        return plt.bar(col_names, self.theta[1:])

# %%
class LassoPenalty:
    
    def __init__(self, l):
        self.l = l # lambda value
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.abs(theta))
        
    def derivation(self, theta):
        return self.l * np.sign(theta)
    
class RidgePenalty:
    
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta
    
class ElasticPenalty:
    
    def __init__(self, l = 0.1, l_ratio = 0.5):
        self.l = l 
        self.l_ratio = l_ratio

    def __call__(self, theta):  #__call__ allows us to call class as method
        l1_contribution = self.l_ratio * self.l * np.sum(np.abs(theta))
        l2_contribution = (1 - self.l_ratio) * self.l * 0.5 * np.sum(np.square(theta))
        return (l1_contribution + l2_contribution)

    def derivation(self, theta):
        l1_derivation = self.l * self.l_ratio * np.sign(theta)
        l2_derivation = self.l * (1 - self.l_ratio) * theta
        return (l1_derivation + l2_derivation)
    
class Lasso(LinearRegression):
    
    def __init__(self, poly, momentum, method, initial_method, lr, l=0.01):
        self.regularization = LassoPenalty(l)
        super().__init__(self.regularization, poly, momentum, method, initial_method, lr)
        
class Ridge(LinearRegression):
    
    def __init__(self, poly, momentum, method, initial_method, lr, l=0.001):
        self.regularization = RidgePenalty(l)
        super().__init__(self.regularization, poly, momentum, method, initial_method, lr)
        
class ElasticNet(LinearRegression):
    
    def __init__(self, poly, momentum, method, initial_method, lr, l=0.005,  l_ratio=0.1):
        self.regularization = ElasticPenalty(l, l_ratio)
        super().__init__(self.regularization, poly, momentum, method, initial_method, lr)

# %%
import sys

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


# %%
from sklearn.model_selection import ParameterGrid

# params_candidate = {'poly':[True,False],
#                 'momentum':[0,0.1],
#                 'method':['batch','mini','sto'],
#                 'initial_method':['zero','xavier'],
#                 'lr':[0.01,0.001,0.0001]}
params_candidate = {'poly':[False],
                'momentum':[0,0.1],
                'method':['mini'],
                'initial_method':['zero','xavier'],
                'lr':[0.01,0.001,0.0001]}
param_grid = ParameterGrid(params_candidate)

best_mse_mse, best_mse_r2, best_r2_mse, best_r2_r2 = np.inf, -np.inf, np.inf, -np.inf

for params in param_grid:
    
    # if params['poly']:
    #     X_train, X_test = X_train_poly, X_test_poly
    #     regs = ['Lasso']
    # else:
    #     X_train, X_test = X_train_lin, X_test_lin
    #     regs = ['Lasso','Ridge','ElasticNet']
    X_train, X_test = X_train_lin, X_test_lin
    regs = ['Lasso']

    
    for reg in regs: 
        mlflow.start_run(run_name=f"method-{params['method']}-lr-{params['lr']}-reg-{reg}-poly-{params['poly']}-momentum-{params['momentum']}-ini-{params['initial_method']}", nested=True)
        
        print("="*5, reg, "="*5)
        print(params)

        # #######
        type_of_regression = str_to_class(reg)    #Ridge, Lasso, ElasticNet
        model = type_of_regression(**params)  
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)

        mse  = model.mse(y_test, np.exp(yhat))
        r2_score = model.r2_score(y_test, np.exp(yhat))

        if mse < best_mse_mse:
            best_mse_model = model
            best_mse_mse = mse
            best_mse_r2 = r2_score
        if r2_score > best_r2_r2:
            best_r2_model = model
            best_r2_mse = mse
            best_r2_r2 = r2_score

        print("Test MSE: ", mse)
        mlflow.log_metric(key="test_mse", value=mse)
        print("Test r2_score: ", r2_score)
        mlflow.log_metric(key="test_r2_score", value=r2_score)

        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, artifact_path='model', signature=signature)

        # #######

        mlflow.end_run()

# %%
print("Test MSE: ", best_r2_mse)
print("Test r2_score: ", best_r2_r2)
print(best_r2_model._coef())
best_r2_model._feature_importance()

print("Test MSE: ", best_r2_mse)
print("Test r2_score: ", best_r2_r2)
print(best_mse_model._coef())
best_r2_model._feature_importance()


# %% [markdown]
# ### Cross Validation & Randomized Search

# file_name = './model/rf_random_selling_price.model'
# pickle.dump(rf_random, open(file_name, 'wb'))

