# 🧠 Neural Networks from Scratch

Building neural networks from first principles using only NumPy. Understand what happens inside BERT, GPT, and modern AI systems.

---

## 🎯 What You'll Learn

- How neural networks actually learn (backpropagation & gradient descent)
- Why depth matters (single neuron vs multi-layer networks)
- The math behind forward and backward propagation
- Connection to transformers and production models

---

## 📊 Course Architecture
```
Blocks 1-2: Single Neuron
           ↓
Blocks 3-4: Gradient Descent & Training
           ↓
Blocks 5-6: Multi-Layer Networks
           ↓
Blocks 7-9: Real Dataset (Digits)
           ↓
Blocks 10-12: Production & Transformers
```

---

## 🔑 Core Concepts

### **1. The Neuron**
```
z = (x1 × w1) + (x2 × w2) + bias    # Weighted sum
output = activation(z)               # Non-linearity
```

Every neuron in BERT does this calculation.

---

### **2. Forward Propagation**
```
Input → Layer 1 → Layer 2 → Layer 3 → Output
  x   →   (W1)  →   (W2)  →   (W3)  →   ŷ
```

Data flows forward through weighted connections and activations.

---

### **3. Backpropagation**
```
Forward:  Input → Layer1 → Layer2 → Output → Loss
Backward: Error → ∂L/∂W2 → ∂L/∂W1 (chain rule)
```

Calculate how much each weight contributed to the error.

---

### **4. Gradient Descent**
```
W_new = W_old - (learning_rate × gradient)
```

Update weights to minimize error. Repeat for many epochs.

**Key hyperparameters:**
- Learning rate too high → explodes
- Learning rate too low → slow convergence
- Learning rate just right → steady progress

---

### **5. Why Multiple Layers?**
```
Single Neuron:
- Decision boundary: ————— (straight line)
- Accuracy: ~85%

Multi-Layer:
- Decision boundary: ∿∿∿∿∿ (curved)
- Accuracy: ~95%
```

**Math:** Stacking layers with activations creates non-linearity:
```
Single: f(x) = w×x + b                    (linear)
Multi:  f(x) = σ(W3×σ(W2×σ(W1×x)))       (non-linear)
```

**Hierarchy:**
```
Layer 1 → Edges, corners
Layer 2 → Shapes, textures  
Layer 3 → Objects, concepts
```

---

### **6. Activation Functions**

| Function | Formula | Use Case |
|----------|---------|----------|
| **ReLU** | `max(0, z)` | Hidden layers (fast, effective) |
| **Sigmoid** | `1/(1+e^(-z))` | Binary output (0-1 probability) |
| **Softmax** | `e^zi / Σe^zj` | Multi-class (probabilities sum to 1) |

**Why needed:** Without activations, multiple layers = still just linear algebra.

---

### **7. Training Loop**
```python
for epoch in range(epochs):
    predictions = forward(X)           # Make predictions
    loss = compute_loss(y, predictions) # Measure error
    gradients = backward(X, y)          # Calculate gradients
    update_weights(gradients)           # Update parameters
```

**Epoch** = One complete pass through dataset

---

### **8. Loss Functions**
```python
# Binary Cross-Entropy
loss = -[y×log(ŷ) + (1-y)×log(1-ŷ)]

# Categorical Cross-Entropy (multi-class)
loss = -Σ(y_true × log(y_pred))
```

Goal: Minimize loss → Better predictions

---

## 🔗 Production Connection

### **Your Network vs BERT**

| Component | Your Network | BERT |
|-----------|-------------|------|
| Input | 64 features | 512 tokens |
| Architecture | 64→128→64→10 | 768→768 (×12 layers) |
| Parameters | ~17K | 110M |
| Task | Digit classification | Language understanding |
| Core concepts | Forward, backward, gradient descent | **Same!** |

### **What Happens in `embedding_model.encode()`**
```
"contract text"
    ↓ Tokenization
[101, 2023, 3820, ...]
    ↓ Embedding Layer
768-dim vectors per token
    ↓ 12 Transformer Layers (your network × 12!)
    ├─ Attention (focus on relevant words)
    └─ Feed-forward (what you built!)
    ↓ Pooling
Final 768-dim embedding
```

---

## 📈 Key Results

**Progression:**
- Single neuron: 50% → 85% accuracy (straight line boundary)
- Multi-layer: 50% → 95%+ accuracy (curved boundary)
- Deep network: 95%+ on handwritten digits (17K parameters)

**Overfitting Detection:**
```
Good:      Train=95%, Test=92%  ✓
Overfit:   Train=99%, Test=75%  ✗ (memorized training data)
Underfit:  Train=60%, Test=58%  ✗ (too simple)
```

---

## 🛠️ Tech Stack
```
Language: Python 3.9+
Core: NumPy (build from scratch)
Visualization: Matplotlib, Seaborn
Data: scikit-learn datasets
Environment: Jupyter Notebook
```

---

## 🚀 Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# Launch notebook
jupyter notebook
```

---

## 📝 Course Structure

**Foundations (Blocks 1-4)**
- Single neuron (perceptron)
- Gradient descent
- Training visualization

**Deep Learning (Blocks 5-6)**
- Multi-layer networks
- Non-linear boundaries

**Real Problem (Blocks 7-9)**
- MNIST digits dataset
- Deep network (3 layers, 17K params)
- Performance analysis

**Production ML (Blocks 10-12)**
- Connection to transformers
- Regularization & optimization
- Final assessment

---

## 💡 Key Takeaways

1. **All neural networks use the same loop:** forward → loss → backward → update
2. **BERT/GPT = Your network × bigger:** Same backprop, just more layers/params
3. **Embeddings = layer outputs:** 768-dim vectors are literally neuron outputs
4. **Production debugging uses same principles:** Check loss curves, learning rate, overfitting

---

## 📚 Next Steps

- Attention mechanisms (transformers)
- CNNs (computer vision)
- RNNs/LSTMs (sequences)
- Advanced optimization (Adam, learning rate schedules)
- Model deployment

---

## 🎓 Prerequisites

- Basic Python
- Linear algebra (matrix multiplication)
- Calculus (derivatives, chain rule - will be explained)
- No prior ML experience needed!

---

**Built with ❤️ for understanding AI from first principles**