import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

# Load everything
X_test = np.load("test_features.npy")
y_test = np.load("test_labels.npy")
svm = joblib.load("svm_sign_language_model.pkl")
le = joblib.load("label_encoder.pkl")

# Predict
y_pred = svm.predict(X_test)

# Classification report
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_, fmt="d")
plt.title("Confusion Matrix for Test Set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix_testset.png", dpi=300)
#plt.show()

# Save predictions to CSV
df = pd.DataFrame({
    "True": le.inverse_transform(y_test),
    "Predicted": le.inverse_transform(y_pred)
})
df.to_csv("svm_predictions.csv", index=False)

confused_pairs = []
for true, pred in zip(y_test, y_pred):
    if true != pred:
        confused_pairs.append((le.inverse_transform([true])[0],
                               le.inverse_transform([pred])[0]))

counter = Counter(confused_pairs)
print("\nMost Confused Sign Pairs:")
for (true, pred), count in counter.most_common(10):
    print(f"'{true}' → '{pred}'  ({count} times)")
