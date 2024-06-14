from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas


mnist = fetch_openml('mnist_784', version=1)
df = mnist


X, y = df['data'], df['target']

# Convert labels to integers
y = y.astype(np.int8)

train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

# train is now 75% of the entire data set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio)

# test is now 10% of the initial data set
# validation is now 15% of the initial data set
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio),random_state=21)


mlp = MLPClassifier(hidden_layer_sizes=(500,500),solver = "adam",learning_rate = "adaptive", early_stopping = True, activation='tanh',batch_size = 250)
def apply_dropout(X, dropout_rate=0.5):
    mask = np.random.binomial(1, 1 - dropout_rate, size=X.shape)
    return X * mask

mlp.fit(x_train, y_train)

# Make predictions on the test data
y_pred = mlp.predict(x_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Create lists to store training and validation loss
train_loss = []
val_loss = []

# Define the number of epochs
epochs = 15

for epoch in range(epochs):
    x_train = apply_dropout(x_train, 0.5)

    # Fit the model for one epoch
    mlp.fit(x_train, y_train)

    # Record training loss
    train_loss.append(mlp.loss_)

    # Predict on the validation set and compute loss
    val_predictions = mlp.predict_proba(x_val)
    val_loss_epoch = -np.mean(np.log(val_predictions[np.arange(len(y_val)), y_val]))
    val_loss.append(val_loss_epoch)

    print(f"Epoch {epoch + 1}/{epochs} - Training loss: {mlp.loss_:.4f}, Validation loss: {val_loss_epoch:.4f}")

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f"Accuracy: {accuracy:.3f} + Batch Normalization")
plt.legend()
plt.show()

