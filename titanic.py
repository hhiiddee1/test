import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import scipy
import numpy as np
import joblib
import dill
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import shap

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],axis=1)
df_test_id = pd.DataFrame(index = df_test['PassengerId'])
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

mlp = MLPClassifier(hidden_layer_sizes=(8,8), activation='relu', solver='adam', max_iter=1000)
mlp.fit(X_train,y_train)

joblib.dump(mlp, 'model/model.joblib')
mlp2 = joblib.load('model/model.joblib')


test_data =[]
for i in X_test_set:
    test_data.append(list(i))
    
f = lambda test_data: mlp.predict(X_train)
print(f)
med=df.median()
print(med)
explainer = shap.KernelExplainer(mlp.predict, X_train)

with open('explainer/explainer.pkl', 'wb') as f:
    dill.dump(explainer, f)
    
with open('explainer/explainer.pkl', 'rb') as f:
    explainer2 = dill.load(f)

# print(test_data)
predict_train = mlp2.predict([list(X_train[1])])
predict_test = mlp2.predict([list(X_test[1])])
predict_test_set = mlp2.predict(test_data)
print(test_data[0])
explainer_train = explainer.shap_values(X_test[1])
print(predict_test_set)
print(explainer_train)
df_test_id['Survived'] = 0
# df_test_id['Survived'] = predict_test_set
df_test_id.to_csv('predictions.csv')
print(df_test_id)