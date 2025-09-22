# Dlt_week-2
Implement a classifier using open-source dataset

1. Import Libraries
We import:
torch and torch.nn for model building
torch.optim for optimization.
scikit-learn for dataset, preprocessing, and splitting.
matplotlib & seaborn for visualization.

2. Load Dataset
We load the Iris dataset from scikit-learn.
It has 150 samples, each with 4 features (sepal length, sepal width, petal length, petal width).
The target variable has 3 classes (Setosa, Versicolor, Virginica).

3. Data Preprocessing
Split into training and testing sets (e.g., 70%-30%).
Scale features using StandardScaler (important for neural networks).
Convert data to PyTorch tensors.

4. Define the Model
We define a Logistic Regression classifier in PyTorch using nn.Linear.
Input size = 4 (features)
Output size = 3 (classes)
This is equivalent to softmax regression.

5. Define Loss & Optimizer
Loss function: nn.CrossEntropyLoss() (suitable for multi-class classification).
Optimizer: Adam (optim.Adam) with learning rate = 0.01.

6. Training Loop
For a fixed number of epochs (e.g., 200):
Forward pass → compute predictions.
Compute loss using criterion.
Backward pass → compute gradients with loss.backward().
Update weights with optimizer.step().

7. Evaluation
Use the trained model to predict on the test set.
Compare predictions with true labels.
Compute accuracy, classification report, and confusion matrix.

8. Visualization
Plot the confusion matrix heatmap to see classification performance.
