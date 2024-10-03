import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=7, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Confusion matrix and classification report:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

with open("../../models/model.pkl", "wb") as file:
    pickle.dump(model, file)
    print("Model saved")
