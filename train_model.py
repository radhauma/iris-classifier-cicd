# train_model.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the Iris dataset from scikit-learn
iris = load_iris()
X, y = iris.data, iris.target  # X: features, y: target classes

# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize a Random Forest Classifier with 100 trees
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Save the trained model to a file using joblib
joblib.dump(clf, "iris_model.pkl")

print("✅ Model trained and saved as iris_model.pkl")
