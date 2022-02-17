from sklearn.neural_network import MLPClassifier
import joblib

x = [[0.0, 0.0, 1.0, -0.6409383911408225, 0.0, 0.0, 0.0, 1.0, 0.0, -0.5921480257661339, 0.0]]

mlp2 = joblib.load('model/model.joblib')
print(mlp2)
predict_test_set = mlp2.predict(x)
print(predict_test_set)