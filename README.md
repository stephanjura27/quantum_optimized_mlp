# Quantum-Inspired Weight Optimization
## Introduction
This is the implementation of the algorithm proposed in the paper called "_Quantum-Inspired Discrete Weight Optimization for Neural Networks Using Grover’s Algorithm_". The algorithm focuses on improving the convergence and accuracy of a neural network (e.g., an MLP). The algorithm starts by initializing the weights of the newtwork from an uniform distribution with a wide spread. Each weight is positioned in a search region around its initial value sampled from the same distribution as a finite set. With a slight variation of Grover’s algorithm, we locate the weights that attain the minimal loss. When the least loss is determined, the weights are then adjusted and the search range is refined according to the loss. This iterative process ensures that the weights are improved while the probability of getting trapped in local minima is minimized.

##Installation
Requirements: Python 3.8+, numpy, matplotlib, scikit-learn, qiskit 0.43.2
First install the requirements, i.e. 

```bash
pip install -r requirements.txt
