# Quantum-Inspired Weight Optimization
## Introduction
This is the implementation of the algorithm proposed in the paper called "_Quantum-Enhanced Discrete Weight Optimization for Neural Networks Using Grover’s Algorithm_". The algorithm focuses on improving the convergence and accuracy of a neural network (e.g., an MLP). The algorithm starts by initializing the weights of the newtwork from an uniform distribution with a wide spread. Each weight is positioned in a search region around its initial value sampled from the same distribution as a finite set. With a slight variation of Grover’s algorithm, we locate the weights that attain the minimal loss. When the least loss is determined, the weights are then adjusted and the search range is refined according to the loss. This iterative process ensures that the weights are improved while the probability of getting trapped in local minima is minimized.

## Installation
Requirements: Ubuntu 20+, Python 3.8+, numpy, matplotlib, scikit-learn, torch, qiskit 0.43.2 

First install the Python packages. Type
```bash
sudo apt install python3-pip -y
```
in the Terminal.

Install the required libraries. 

```bash
pip install numpy scikit-learn matplotlib torch qiskit==0.43.2
```
As an alternative, you can use Google Colaboratory. Numpy, matplotlib, scikit-learn are already installed there, so you need to install qiskit.
```bash
pip install qiskit==0.43.2
```

On Ubuntu, run the main.py file for training the quantum-optimized MLP, classical.py for training an MLP trained using gradient descent backpropagation algorithm with ADAM optimizer or DeepNetwork.py for the deep quantum-optimized network.
```bash
python3 main.py
```
```bash
python3 classical.py
```

On Google Colaboratory, copy the code from the desired file in a cell and then run it. 

If you want to replace the cross-entropy loss with the non-convex function presented in the paper, simply replace the formula from the code with
```python
loss = alpha * (-torch.sum(targets * torch.log(outputs + 1e-9), dim=1)) + beta * torch.sum(torch.sin(phi * outputs), dim=1)
```

