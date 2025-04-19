import numpy as np
from sklearn.datasets import load_digits
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

# Rețea cu 3 straturi ascunse: input -> hidden1 -> hidden2 -> hidden3 -> output
class MLP3HL:
    def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size, output_size, dropout_rate=0.0):
        np.random.seed(42)
        self.weights_1 = np.random.uniform(-1, 1, (input_size, hidden1_size))
        self.weights_2 = np.random.uniform(-1, 1, (hidden1_size, hidden2_size))
        self.weights_3 = np.random.uniform(-1, 1, (hidden2_size, hidden3_size))
        self.weights_4 = np.random.uniform(-1, 1, (hidden3_size, output_size))
        self.dropout_rate = dropout_rate

    def forward(self, x, training=True):
        hidden1 = np.tanh(np.dot(x, self.weights_1))
        if training and self.dropout_rate > 0:
            mask = (np.random.rand(*hidden1.shape) > self.dropout_rate).astype(float)
            hidden1 *= mask
            hidden1 /= (1.0 - self.dropout_rate)
        hidden2 = np.tanh(np.dot(hidden1, self.weights_2))
        if training and self.dropout_rate > 0:
            mask = (np.random.rand(*hidden2.shape) > self.dropout_rate).astype(float)
            hidden2 *= mask
            hidden2 /= (1.0 - self.dropout_rate)
        hidden3 = np.tanh(np.dot(hidden2, self.weights_3))
        if training and self.dropout_rate > 0:
            mask = (np.random.rand(*hidden3.shape) > self.dropout_rate).astype(float)
            hidden3 *= mask
            hidden3 /= (1.0 - self.dropout_rate)
        output = self.softmax(np.dot(hidden3, self.weights_4))
        return hidden1, hidden2, hidden3, output

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Funcții de forward cu greutăți custom pentru fiecare strat
def forward_with_custom_hidden1(mlp, x, custom_hidden1_weights, training=False):
    hidden1 = np.tanh(np.dot(x, custom_hidden1_weights))
    if training and mlp.dropout_rate > 0:
        mask = (np.random.rand(*hidden1.shape) > mlp.dropout_rate).astype(float)
        hidden1 *= mask
        hidden1 /= (1.0 - mlp.dropout_rate)
    hidden2 = np.tanh(np.dot(hidden1, mlp.weights_2))
    if training and mlp.dropout_rate > 0:
        mask = (np.random.rand(*hidden2.shape) > mlp.dropout_rate).astype(float)
        hidden2 *= mask
        hidden2 /= (1.0 - mlp.dropout_rate)
    hidden3 = np.tanh(np.dot(hidden2, mlp.weights_3))
    if training and mlp.dropout_rate > 0:
        mask = (np.random.rand(*hidden3.shape) > mlp.dropout_rate).astype(float)
        hidden3 *= mask
        hidden3 /= (1.0 - mlp.dropout_rate)
    output = mlp.softmax(np.dot(hidden3, mlp.weights_4))
    return hidden1, hidden2, hidden3, output

def forward_with_custom_hidden2(mlp, hidden1, custom_hidden2_weights, training=False):
    hidden2 = np.tanh(np.dot(hidden1, custom_hidden2_weights))
    if training and mlp.dropout_rate > 0:
        mask = (np.random.rand(*hidden2.shape) > mlp.dropout_rate).astype(float)
        hidden2 *= mask
        hidden2 /= (1.0 - mlp.dropout_rate)
    hidden3 = np.tanh(np.dot(hidden2, mlp.weights_3))
    if training and mlp.dropout_rate > 0:
        mask = (np.random.rand(*hidden3.shape) > mlp.dropout_rate).astype(float)
        hidden3 *= mask
        hidden3 /= (1.0 - mlp.dropout_rate)
    output = mlp.softmax(np.dot(hidden3, mlp.weights_4))
    return hidden2, hidden3, output

def forward_with_custom_hidden3(mlp, hidden2, custom_hidden3_weights, training=False):
    hidden3 = np.tanh(np.dot(hidden2, custom_hidden3_weights))
    if training and mlp.dropout_rate > 0:
        mask = (np.random.rand(*hidden3.shape) > mlp.dropout_rate).astype(float)
        hidden3 *= mask
        hidden3 /= (1.0 - mlp.dropout_rate)
    output = mlp.softmax(np.dot(hidden3, mlp.weights_4))
    return hidden3, output

def forward_with_custom_output(mlp, hidden3, custom_output_weights, training=False):
    output = mlp.softmax(np.dot(hidden3, custom_output_weights))
    return hidden3, output

# Funcția de optimizare a greutăților cu Grover, extinsă pentru straturile "hidden1", "hidden2", "hidden3" și "output"
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
                if layer_type == "hidden1":
                    _, _, _, temp_output = forward_with_custom_hidden1(mlp, layer_input, temp_weights, training=False)
                    reg = weight_decay * (np.sum(temp_weights**2) + np.sum(mlp.weights_2**2) +
                                          np.sum(mlp.weights_3**2) + np.sum(mlp.weights_4**2))
                elif layer_type == "hidden2":
                    # layer_input reprezintă output-ul din hidden1
                    _, _, temp_output = forward_with_custom_hidden2(mlp, layer_input, temp_weights, training=False)
                    reg = weight_decay * (np.sum(mlp.weights_1**2) + np.sum(temp_weights**2) +
                                          np.sum(mlp.weights_3**2) + np.sum(mlp.weights_4**2))
                elif layer_type == "hidden3":
                    # layer_input reprezintă output-ul din hidden2
                    _, temp_output = forward_with_custom_hidden3(mlp, layer_input, temp_weights, training=False)
                    reg = weight_decay * (np.sum(mlp.weights_1**2) + np.sum(mlp.weights_2**2) +
                                          np.sum(temp_weights**2) + np.sum(mlp.weights_4**2))
                elif layer_type == "output":
                    # layer_input reprezintă output-ul din hidden3
                    _, temp_output = forward_with_custom_output(mlp, layer_input, temp_weights, training=False)
                    reg = weight_decay * (np.sum(mlp.weights_1**2) + np.sum(mlp.weights_2**2) +
                                          np.sum(mlp.weights_3**2) + np.sum(temp_weights**2))
                else:
                    raise ValueError("Invalid layer type.")
                ce_loss = -np.mean(np.sum(y_true * np.log(temp_output + 1e-9), axis=1))
                total_loss = ce_loss + reg
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
            if layer_type == "hidden1":
                _, _, _, temp_output = forward_with_custom_hidden1(mlp, layer_input, temp_weights, training=False)
                reg = weight_decay * (np.sum(temp_weights**2) + np.sum(mlp.weights_2**2) +
                                      np.sum(mlp.weights_3**2) + np.sum(mlp.weights_4**2))
                _, _, _, current_output = forward_with_custom_hidden1(mlp, layer_input, optimized_weights, training=False)
                current_reg = weight_decay * (np.sum(optimized_weights**2) + np.sum(mlp.weights_2**2) +
                                              np.sum(mlp.weights_3**2) + np.sum(mlp.weights_4**2))
            elif layer_type == "hidden2":
                _, _, temp_output = forward_with_custom_hidden2(mlp, layer_input, temp_weights, training=False)
                reg = weight_decay * (np.sum(mlp.weights_1**2) + np.sum(temp_weights**2) +
                                      np.sum(mlp.weights_3**2) + np.sum(mlp.weights_4**2))
                _, _, current_output = forward_with_custom_hidden2(mlp, layer_input, optimized_weights, training=False)
                current_reg = weight_decay * (np.sum(mlp.weights_1**2) + np.sum(optimized_weights**2) +
                                              np.sum(mlp.weights_3**2) + np.sum(mlp.weights_4**2))
            elif layer_type == "hidden3":
                _, temp_output = forward_with_custom_hidden3(mlp, layer_input, temp_weights, training=False)
                reg = weight_decay * (np.sum(mlp.weights_1**2) + np.sum(mlp.weights_2**2) +
                                      np.sum(temp_weights**2) + np.sum(mlp.weights_4**2))
                _, current_output = forward_with_custom_hidden3(mlp, layer_input, optimized_weights, training=False)
                current_reg = weight_decay * (np.sum(mlp.weights_1**2) + np.sum(mlp.weights_2**2) +
                                              np.sum(optimized_weights**2) + np.sum(mlp.weights_4**2))
            elif layer_type == "output":
                _, temp_output = forward_with_custom_output(mlp, layer_input, temp_weights, training=False)
                reg = weight_decay * (np.sum(mlp.weights_1**2) + np.sum(mlp.weights_2**2) +
                                      np.sum(mlp.weights_3**2) + np.sum(temp_weights**2))
                _, current_output = forward_with_custom_output(mlp, layer_input, optimized_weights, training=False)
                current_reg = weight_decay * (np.sum(mlp.weights_1**2) + np.sum(mlp.weights_2**2) +
                                              np.sum(mlp.weights_3**2) + np.sum(optimized_weights**2))
            new_ce_loss = -np.mean(np.sum(y_true * np.log(temp_output + 1e-9), axis=1))
            new_loss = new_ce_loss + reg
            current_ce_loss = -np.mean(np.sum(y_true * np.log(current_output + 1e-9), axis=1))
            current_loss = current_ce_loss + current_reg
            if new_loss < current_loss:
                optimized_weights[i, j] = proposed_weight
    return optimized_weights

if __name__ == "__main__":
    # Se încarcă dataset-ul digits
    data = load_digits()
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    # Normalizare cu adăugarea unui epsilon pentru a evita diviziunea cu zero
    epsilon = 1e-9
    x_train = (x_train - np.mean(x_train, axis=0)) / (np.std(x_train, axis=0) + epsilon)
    x_test  = (x_test  - np.mean(x_test, axis=0)) / (np.std(x_test, axis=0) + epsilon)
    
    encoder = OneHotEncoder(sparse_output=False)
    y_train_1hot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_1hot  = encoder.transform(y_test.reshape(-1, 1))
    
    # Creare rețea cu 3 straturi ascunse
    mlp = MLP3HL(
        input_size=x_train.shape[1],
        hidden1_size=64,
        hidden2_size=32,
        hidden3_size=16,
        output_size=y_train_1hot.shape[1],
        dropout_rate=0.2
    )
    
    num_epochs = 10
    resolution_hidden1 = 32
    resolution_hidden2 = 32
    resolution_hidden3 = 32
    resolution_output  = 64
    initial_search_std_factor = 0.5
    weight_decay = 1e-3
    # Factori de ajustare pentru fiecare strat
    search_std_factor_hidden1 = initial_search_std_factor
    search_std_factor_hidden2 = initial_search_std_factor
    search_std_factor_hidden3 = initial_search_std_factor
    search_std_factor_output  = initial_search_std_factor
    min_search_std = 0.1
    max_search_std = 2.0
    tol_ratio_hidden1 = 0.05
    tol_ratio_hidden2 = 0.05
    tol_ratio_hidden3 = 0.05
    tol_ratio_output  = 0.1
    
    train_acc_all, test_acc_all = [], []
    train_loss_all, test_loss_all = [], []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        # Evaluare inițială
        hidden1_before, hidden2_before, hidden3_before, output_before = mlp.forward(x_train, training=False)
        ce_train_before = -np.mean(np.sum(y_train_1hot * np.log(output_before + 1e-9), axis=1))
        reg_train_before = weight_decay * (np.sum(mlp.weights_1**2) + np.sum(mlp.weights_2**2) +
                                           np.sum(mlp.weights_3**2) + np.sum(mlp.weights_4**2))
        loss_train_before = ce_train_before + reg_train_before
        
        # Optimizare pentru hidden1
        mlp.weights_1 = optimize_weights_with_grover(
            mlp=mlp,
            layer_input=x_train,
            y_true=y_train_1hot,
            weights=mlp.weights_1,
            layer_type="hidden1",
            resolution=resolution_hidden1,
            search_std_factor=search_std_factor_hidden1,
            weight_decay=weight_decay,
            tol_ratio=tol_ratio_hidden1
        )
        hidden1_after, hidden2_after, hidden3_after, output_after = mlp.forward(x_train, training=False)
        ce_train_after_hidden1 = -np.mean(np.sum(y_train_1hot * np.log(output_after + 1e-9), axis=1))
        reg_train_after_hidden1 = weight_decay * (np.sum(mlp.weights_1**2) + np.sum(mlp.weights_2**2) +
                                                  np.sum(mlp.weights_3**2) + np.sum(mlp.weights_4**2))
        loss_train_after_hidden1 = ce_train_after_hidden1 + reg_train_after_hidden1
        if loss_train_after_hidden1 < loss_train_before:
            search_std_factor_hidden1 = max(search_std_factor_hidden1 * 0.95, min_search_std)
        else:
            search_std_factor_hidden1 = min(search_std_factor_hidden1 * 1.05, max_search_std)
        
        # Pentru hidden2, folosim output-ul din hidden1 actualizat
        hidden1_updated, _, _, _ = mlp.forward(x_train, training=False)
        mlp.weights_2 = optimize_weights_with_grover(
            mlp=mlp,
            layer_input=hidden1_updated,
            y_true=y_train_1hot,
            weights=mlp.weights_2,
            layer_type="hidden2",
            resolution=resolution_hidden2,
            search_std_factor=search_std_factor_hidden2,
            weight_decay=weight_decay,
            tol_ratio=tol_ratio_hidden2
        )
        _, hidden2_after2, hidden3_after2, output_after_hidden2 = mlp.forward(x_train, training=False)
        ce_train_after_hidden2 = -np.mean(np.sum(y_train_1hot * np.log(output_after_hidden2 + 1e-9), axis=1))
        reg_train_after_hidden2 = weight_decay * (np.sum(mlp.weights_1**2) + np.sum(mlp.weights_2**2) +
                                                   np.sum(mlp.weights_3**2) + np.sum(mlp.weights_4**2))
        loss_train_after_hidden2 = ce_train_after_hidden2 + reg_train_after_hidden2
        if loss_train_after_hidden2 < loss_train_after_hidden1:
            search_std_factor_hidden2 = max(search_std_factor_hidden2 * 0.95, min_search_std)
        else:
            search_std_factor_hidden2 = min(search_std_factor_hidden2 * 1.05, max_search_std)
        
        # Pentru hidden3, folosim output-ul din hidden2 actualizat
        _, hidden2_updated, _, _ = mlp.forward(x_train, training=False)
        mlp.weights_3 = optimize_weights_with_grover(
            mlp=mlp,
            layer_input=hidden2_updated,
            y_true=y_train_1hot,
            weights=mlp.weights_3,
            layer_type="hidden3",
            resolution=resolution_hidden3,
            search_std_factor=search_std_factor_hidden3,
            weight_decay=weight_decay,
            tol_ratio=tol_ratio_hidden3
        )
        _, _, hidden3_after3, output_after_hidden3 = mlp.forward(x_train, training=False)
        ce_train_after_hidden3 = -np.mean(np.sum(y_train_1hot * np.log(output_after_hidden3 + 1e-9), axis=1))
        reg_train_after_hidden3 = weight_decay * (np.sum(mlp.weights_1**2) + np.sum(mlp.weights_2**2) +
                                                   np.sum(mlp.weights_3**2) + np.sum(mlp.weights_4**2))
        loss_train_after_hidden3 = ce_train_after_hidden3 + reg_train_after_hidden3
        if loss_train_after_hidden3 < loss_train_after_hidden2:
            search_std_factor_hidden3 = max(search_std_factor_hidden3 * 0.95, min_search_std)
        else:
            search_std_factor_hidden3 = min(search_std_factor_hidden3 * 1.05, max_search_std)
        
        # Pentru stratul de output, folosim output-ul din hidden3 actualizat
        _, _, hidden3_updated, _ = mlp.forward(x_train, training=False)
        mlp.weights_4 = optimize_weights_with_grover(
            mlp=mlp,
            layer_input=hidden3_updated,
            y_true=y_train_1hot,
            weights=mlp.weights_4,
            layer_type="output",
            resolution=resolution_output,
            search_std_factor=search_std_factor_output,
            weight_decay=weight_decay,
            tol_ratio=tol_ratio_output
        )
        _, output_after_all = forward_with_custom_output(mlp, hidden3_updated, mlp.weights_4, training=False)
        ce_train_after_output = -np.mean(np.sum(y_train_1hot * np.log(output_after_all + 1e-9), axis=1))
        reg_train_after_output = weight_decay * (np.sum(mlp.weights_1**2) + np.sum(mlp.weights_2**2) +
                                                  np.sum(mlp.weights_3**2) + np.sum(mlp.weights_4**2))
        loss_train_after_output = ce_train_after_output + reg_train_after_output
        if loss_train_after_output < loss_train_after_hidden3:
            search_std_factor_output = max(search_std_factor_output * 0.95, min_search_std)
        else:
            search_std_factor_output = min(search_std_factor_output * 1.05, max_search_std)
        
        # Evaluare finală pe datele de antrenare și testare
        _, _, _, y_train_pred = mlp.forward(x_train, training=False)
        ce_train = -np.mean(np.sum(y_train_1hot * np.log(y_train_pred + 1e-9), axis=1))
        reg_train = weight_decay * (np.sum(mlp.weights_1**2) + np.sum(mlp.weights_2**2) +
                                    np.sum(mlp.weights_3**2) + np.sum(mlp.weights_4**2))
        loss_train = ce_train + reg_train
        train_acc = (np.argmax(y_train_pred, axis=1) == y_train).mean() * 100
        
        _, _, _, y_test_pred = mlp.forward(x_test, training=False)
        ce_test = -np.mean(np.sum(y_test_1hot * np.log(y_test_pred + 1e-9), axis=1))
        reg_test = weight_decay * (np.sum(mlp.weights_1**2) + np.sum(mlp.weights_2**2) +
                                   np.sum(mlp.weights_3**2) + np.sum(mlp.weights_4**2))
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
    plt.title('Accuracy over Epochs (Digits dataset)')
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_loss_all, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_loss_all, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs (Digits dataset)')
    plt.tight_layout()
    plt.show()