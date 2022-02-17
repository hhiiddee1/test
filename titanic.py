import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import numpy as np
sns.set()

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],axis=1)
df_test_id = pd.DataFrame()
df_test_id.index = df_test['PassengerId']
df_test = df_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],axis=1)

df['Sex'] = pd.factorize(df['Sex'])[0]
df_test['Sex'] = pd.factorize(df_test['Sex'])[0]

pd.get_dummies(df, prefix=['Embarked'])

df['Pclass'] = df['Pclass'].apply(str)
df = pd.get_dummies(df, prefix=['PClass','Embarked'])
df_test['Pclass'] = df_test['Pclass'].apply(str)
df_test = pd.get_dummies(df_test, prefix=['PClass','Embarked'])

df['z_score']=scipy.stats.zscore(df['Fare'])
df.loc[df['z_score'].abs()>=3, 'Fare'] = df['Fare'].mean()
df = df.drop(['z_score'], axis=1)


df.isnull().sum(axis = 0)
df = df.fillna(df.mean())
df.isnull().sum(axis = 0)

df_test.isnull().sum(axis = 0)
df_test = df_test.fillna(df.mean())
df_test.isnull().sum(axis = 0)

df['Age_id'] = 0
df.loc[df['Age']<= 15,'Age_id'] = '~15'
df.loc[(df['Age'] > 15) & (df['Age'] >= 25),'Age_id'] = '15~25'
df.loc[(df['Age'] > 25) & (df['Age'] >= 35),'Age_id'] = '15~25'
df.loc[(df['Age'] > 35) & (df['Age'] >= 45),'Age_id'] = '15~25'
df.loc[(df['Age'] > 45) & (df['Age'] >= 55),'Age_id'] = '15~25'
df.loc[df['Age'] > 55,'Age_id'] = '55~'

df['Age']=(df['Age']-df['Age'].mean())/df['Age'].std()
df_test['Age']=(df_test['Age']-df_test['Age'].mean())/df_test['Age'].std()
df['Fare']=(df['Fare']-df['Fare'].mean())/df['Fare'].std()
df_test['Fare']=(df_test['Fare']-df_test['Fare'].mean())/df_test['Fare'].std()

target_column = 'Survived'
predictors = list(set(list(df.columns))-set([target_column, 'Age_id']))
predictors_test = list(set(list(df.columns))-set(['Age_id']))


X = df[predictors].values
y = df[target_column].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)

X_test_set =df_test[predictors].values

mlp = MLPClassifier(hidden_layer_sizes=(8,8), activation='relu', solver='adam', max_iter=500)
mlp.fit(X_train,y_train)
print(X_test[1])
predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)
predict_test_set = mlp.predict(X_test_set)
df_test_id['Survived'] = 0
df_test_id['Survived'] = predict_test_set
df_test_id.to_csv('predictions.csv')
print(df_test_id)