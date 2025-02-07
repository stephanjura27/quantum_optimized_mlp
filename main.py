import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, Aer, execute
from math import pi, floor, sqrt

def oracle(num_qubits, marked_states):
    qc = QuantumCircuit(num_qubits + 1)
    for state in marked_states:
        binary = format(state, f'0{num_qubits}b')
        for i, bit in enumerate(reversed(binary)):
            if bit == '0':
                qc.x(i)
        qc.h(num_qubits)
        qc.mcx(list(range(num_qubits)), num_qubits)
        qc.h(num_qubits)
        for i, bit in enumerate(reversed(binary)):
            if bit == '0':
                qc.x(i)
    qc = qc.to_gate()
    qc.name = "Oracle"
    return qc

def diffusion_operator(num_qubits):
    qc = QuantumCircuit(num_qubits + 1)
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits)
    qc.mcx(list(range(num_qubits)), num_qubits)
    qc.h(num_qubits)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    qc = qc.to_gate()
    qc.name = "Diffusion"
    return qc

def quantum_min_partial(num_qubits, marked_states):
    M = len(marked_states)
    N = 2 ** num_qubits
    if M == 0:
        raise ValueError("There are no marked states for Grover.")
    k = floor((pi / 4) * sqrt(N / M))
    k = max(k, 1)
    qc = QuantumCircuit(num_qubits + 1, num_qubits)
    qc.h(range(num_qubits))
    qc.x(num_qubits)
    qc.h(num_qubits)
    oracle_gate = oracle(num_qubits, marked_states)
    diffusion_gate = diffusion_operator(num_qubits)
    for _ in range(k):
        qc.append(oracle_gate, range(num_qubits + 1))
        qc.append(diffusion_gate, range(num_qubits + 1))
    qc.measure(range(num_qubits), range(num_qubits))
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, backend=simulator, shots=1024).result()
    counts = result.get_counts()
    if not counts:
        raise ValueError("No results obtained from simulation.")
    max_state = max(counts, key=counts.get)
    chosen_index = int(max_state, 2)
    if chosen_index >= N:
        raise ValueError("Chosen index exceeds the search space size.")
    return chosen_index

class MLP:
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.0):
        np.random.seed(42)
        self.weights_1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.weights_2 = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.dropout_rate = dropout_rate

    def forward(self, x, training=True):
        hidden = np.tanh(np.dot(x, self.weights_1))
        if training and self.dropout_rate > 0:
            mask = (np.random.rand(*hidden.shape) > self.dropout_rate).astype(float)
            hidden *= mask
            hidden /= (1.0 - self.dropout_rate)
        output = self.softmax(np.dot(hidden, self.weights_2))
        return hidden, output

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward_with_custom_hidden(mlp, x, custom_hidden_weights, output_weights, training=False):
    hidden = np.tanh(np.dot(x, custom_hidden_weights))
    if training and mlp.dropout_rate > 0:
        mask = (np.random.rand(*hidden.shape) > mlp.dropout_rate).astype(float)
        hidden *= mask
        hidden /= (1.0 - mlp.dropout_rate)
    output = mlp.softmax(np.dot(hidden, output_weights))
    return hidden, output

def forward_with_custom_output(mlp, hidden_input, custom_output_weights, training=False):
    if training and mlp.dropout_rate > 0:
        mask = (np.random.rand(*hidden_input.shape) > mlp.dropout_rate).astype(float)
        hidden_input *= mask
        hidden_input /= (1.0 - mlp.dropout_rate)
    output = mlp.softmax(np.dot(hidden_input, custom_output_weights))
    return hidden_input, output

def optimize_weights_with_grover(
    mlp,
    layer_input,
    y_true,
    weights,
    layer_type,
    resolution=32,
    search_std_factor=0.5,
    weight_decay=1e-3,
    tol_ratio=0.05
):
    optimized_weights = weights.copy()
    std_weight = np.std(weights)
    if std_weight < 1e-12:
        std_weight = 1.0
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            current_weight = weights[i, j]
            low  = current_weight - search_std_factor * std_weight
            high = current_weight + search_std_factor * std_weight
            T = np.linspace(low, high, resolution)
            loss_candidates = []
            for candidate_val in T:
                temp_weights = optimized_weights.copy()
                temp_weights[i, j] = candidate_val
                if layer_type == "hidden":
                    _, temp_output = forward_with_custom_hidden(
                        mlp, layer_input, temp_weights, mlp.weights_2, training=False
                    )
                    w1_sum_sqr = np.sum(temp_weights**2)
                    w2_sum_sqr = np.sum(mlp.weights_2**2)
                else:
                    _, temp_output = forward_with_custom_output(
                        mlp, layer_input, temp_weights, training=False
                    )
                    w1_sum_sqr = np.sum(mlp.weights_1**2)
                    w2_sum_sqr = np.sum(temp_weights**2)
                ce_loss = -np.mean(np.sum(y_true * np.log(temp_output + 1e-9), axis=1))
                reg_term = weight_decay * (w1_sum_sqr + w2_sum_sqr)
                total_loss = ce_loss + reg_term
                loss_candidates.append(total_loss)
            min_loss = np.min(loss_candidates)
            tol = tol_ratio * min_loss
            marked_states = [idx for idx, loss in enumerate(loss_candidates) if (loss <= min_loss + tol)]
            if not marked_states:
                idx_min = np.argmin(loss_candidates)
            else:
                try:
                    idx_min = quantum_min_partial(num_qubits=int(np.ceil(np.log2(len(T)))), marked_states=marked_states)
                except Exception:
                    idx_min = np.argmin(loss_candidates)
            if idx_min >= len(T):
                idx_min = np.argmin(loss_candidates)
            proposed_weight = T[idx_min]
            temp_weights = optimized_weights.copy()
            temp_weights[i, j] = proposed_weight
            if layer_type == "hidden":
                _, temp_output = forward_with_custom_hidden(
                    mlp, layer_input, temp_weights, mlp.weights_2, training=False
                )
                w1_sum_sqr = np.sum(temp_weights**2)
                w2_sum_sqr = np.sum(mlp.weights_2**2)
            else:
                _, temp_output = forward_with_custom_output(
                    mlp, layer_input, temp_weights, training=False
                )
                w1_sum_sqr = np.sum(mlp.weights_1**2)
                w2_sum_sqr = np.sum(temp_weights**2)
            new_ce_loss = -np.mean(np.sum(y_true * np.log(temp_output + 1e-9), axis=1))
            new_loss = new_ce_loss + weight_decay * (w1_sum_sqr + w2_sum_sqr)
            if layer_type == "hidden":
                _, current_output = forward_with_custom_hidden(
                    mlp, layer_input, optimized_weights, mlp.weights_2, training=False
                )
                current_ce_loss = -np.mean(np.sum(y_true * np.log(current_output + 1e-9), axis=1))
                current_loss = current_ce_loss + weight_decay * (np.sum(optimized_weights**2) + np.sum(mlp.weights_2**2))
            else:
                _, current_output = forward_with_custom_output(
                    mlp, layer_input, optimized_weights, training=False
                )
                current_ce_loss = -np.mean(np.sum(y_true * np.log(current_output + 1e-9), axis=1))
                current_loss = current_ce_loss + weight_decay * (np.sum(mlp.weights_1**2) + np.sum(optimized_weights**2))
            if new_loss < current_loss:
                optimized_weights[i, j] = proposed_weight
    return optimized_weights

if __name__ == "__main__":
    data = load_wine()
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    x_test  = (x_test  - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)
    encoder = OneHotEncoder(sparse_output=False)
    y_train_1hot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_1hot  = encoder.transform(y_test.reshape(-1, 1))
    mlp = MLP(input_size=13, hidden_size=32, output_size=3, dropout_rate=0.2)
    num_epochs = 10
    resolution_hidden = 32
    resolution_output = 64
    initial_search_std_factor = 0.5
    weight_decay = 1e-3
    search_std_factor_hidden = initial_search_std_factor
    search_std_factor_output = initial_search_std_factor
    min_search_std = 0.1
    max_search_std = 2.0
    tol_ratio_hidden = 0.05
    tol_ratio_output = 0.1
    train_acc_all, test_acc_all = [], []
    train_loss_all, test_loss_all = [], []
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        hidden_before, _ = mlp.forward(x_train, training=False)
        ce_train_before_hidden = -np.mean(np.sum(y_train_1hot * np.log(mlp.softmax(np.dot(hidden_before, mlp.weights_2)) + 1e-9), axis=1))
        reg_train_before_hidden = weight_decay * (np.sum(mlp.weights_1**2) + np.sum(mlp.weights_2**2))
        loss_train_before_hidden = ce_train_before_hidden + reg_train_before_hidden
        mlp.weights_1 = optimize_weights_with_grover(
            mlp=mlp,
            layer_input=x_train,
            y_true=y_train_1hot,
            weights=mlp.weights_1,
            layer_type="hidden",
            resolution=resolution_hidden,
            search_std_factor=search_std_factor_hidden,
            weight_decay=weight_decay,
            tol_ratio=tol_ratio_hidden
        )
        hidden_after, _ = mlp.forward(x_train, training=False)
        ce_train_after_hidden = -np.mean(np.sum(y_train_1hot * np.log(mlp.softmax(np.dot(hidden_after, mlp.weights_2)) + 1e-9), axis=1))
        reg_train_after_hidden = weight_decay * (np.sum(mlp.weights_1**2) + np.sum(mlp.weights_2**2))
        loss_train_after_hidden = ce_train_after_hidden + reg_train_after_hidden
        if loss_train_after_hidden < loss_train_before_hidden:
            search_std_factor_hidden = max(search_std_factor_hidden * 0.95, min_search_std)
        else:
            search_std_factor_hidden = min(search_std_factor_hidden * 1.05, max_search_std)
        hidden_output_after_hidden, _ = mlp.forward(x_train, training=False)
        loss_train_before_output = loss_train_after_hidden
        mlp.weights_2 = optimize_weights_with_grover(
            mlp=mlp,
            layer_input=hidden_output_after_hidden,
            y_true=y_train_1hot,
            weights=mlp.weights_2,
            layer_type="output",
            resolution=resolution_output,
            search_std_factor=search_std_factor_output,
            weight_decay=weight_decay,
            tol_ratio=tol_ratio_output
        )
        _, y_train_pred_after_output = mlp.forward(x_train, training=False)
        ce_train_after_output = -np.mean(np.sum(y_train_1hot * np.log(y_train_pred_after_output + 1e-9), axis=1))
        reg_train_after_output = weight_decay * (np.sum(mlp.weights_1**2) + np.sum(mlp.weights_2**2))
        loss_train_after_output = ce_train_after_output + reg_train_after_output
        if loss_train_after_output < loss_train_before_output:
            search_std_factor_output = max(search_std_factor_output * 0.95, min_search_std)
        else:
            search_std_factor_output = min(search_std_factor_output * 1.05, max_search_std)
        _, y_train_pred = mlp.forward(x_train, training=False)
        ce_train = -np.mean(np.sum(y_train_1hot * np.log(y_train_pred + 1e-9), axis=1))
        reg_train = weight_decay * (np.sum(mlp.weights_1**2) + np.sum(mlp.weights_2**2))
        loss_train = ce_train + reg_train
        train_acc = (np.argmax(y_train_pred, axis=1) == y_train).mean() * 100
        _, y_test_pred = mlp.forward(x_test, training=False)
        ce_test = -np.mean(np.sum(y_test_1hot * np.log(y_test_pred + 1e-9), axis=1))
        reg_test = weight_decay * (np.sum(mlp.weights_1**2) + np.sum(mlp.weights_2**2))
        loss_test = ce_test + reg_test
        test_acc = (np.argmax(y_test_pred, axis=1) == y_test).mean() * 100
        train_acc_all.append(train_acc)
        test_acc_all.append(test_acc)
        train_loss_all.append(loss_train)
        test_loss_all.append(loss_test)
        print(f"  Train Loss: {loss_train:.4f}, Train Acc: {train_acc:.2f}% | Test Loss: {loss_test:.4f}, Test Acc: {test_acc:.2f}%")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_acc_all, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), test_acc_all, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy over Epochs (Wine dataset)')
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_loss_all, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_loss_all, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs (Wine dataset)')
    plt.tight_layout()
    plt.show()
