import numpy as np
from matplotlib import pyplot as plt
np.random.seed(0)
# Generate synthetic data for demonstration
def generate_data(n_samples, n_features, noise=0.1):
    X = np.random.rand(n_samples, n_features)
    true_weights = np.random.rand(n_features)
    y = X @ true_weights + noise * np.random.randn(n_samples)
    return X, y, true_weights

# Define Linear Regression model with MSE and gradient calculation
class LinearRegressionModel:
    def __init__(self, n_features):
        self.weights = np.zeros(n_features)

    def predict(self, X):
        return X @ self.weights

    def compute_gradient(self, X, y):
        predictions = self.predict(X)
        errors = predictions - y
        gradient = 2 * X.T @ errors / len(y)
        return gradient, errors


    def update_weights(self, gradient, lr=0.01):
        self.weights -= lr * gradient


# Simulate a federated learning process
def federated_linear_regression(num_nodes=5, n_samples=1000, n_features=10, rounds=2500, lr=0.01):
    # Generate data and split it across nodes
    myloss = []
    X, y, true_weights = generate_data(n_samples, n_features)
    node_data = [(X[i::num_nodes], y[i::num_nodes]) for i in range(num_nodes)]
    
    # Initialize the global model
    global_model = LinearRegressionModel(n_features)

    # Federated learning rounds
    for round_num in range(rounds):
        #print(f"Round {round_num + 1}/{rounds}")
        local_weights = []
        temp = []
        # Each node trains on its local data and computes its weight update
        for node_id, (X_node, y_node) in enumerate(node_data):
            
            local_model = LinearRegressionModel(n_features)
            local_model.weights = global_model.weights.copy()  # Start with global model weights
            
            # Local training (one gradient descent step for simplicity)
            gradient, loss = local_model.compute_gradient(X_node, y_node)
            temp.append(loss @ loss)
            local_model.update_weights(gradient, lr=lr)
            local_weights.append(local_model.weights)
            #print(f"  Node {node_id + 1}: Updated weights - {local_model.weights}")
        myloss.append(temp)
        # Aggregation (FedAvg): average weights from all nodes
        global_model.weights = np.mean(local_weights, axis=0)
        #print(f"  Updated global model weights: {global_model.weights}\n")

    # Final global model weights and comparison to true weights
    print("Final global model weights:", global_model.weights)
    print("True weights (for reference):", true_weights)
    return global_model.weights, myloss

# Run federated linear regression
final_weights, loss = federated_linear_regression()
start = 0
for k in range(5):
    plt.plot(np.arange(start,len(loss),step=1), [loss[i][k] for i in range(start, len(loss))], label=f'Trainer {k+1}')
    
plt.plot(np.arange(start,len(loss),step=1), )
plt.xscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(visible=True, which='both', axis='both')
plt.legend()
plt.savefig('loss.jpg', dpi=400, bbox_inches='tight')