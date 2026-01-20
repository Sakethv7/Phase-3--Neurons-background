# 🧠 Neural Networks from Scratch - Quick Reference

## 🎯 Core Concepts (What You Built)

### **1. The Neuron**
```
z = (x1 * w1) + (x2 * w2) + bias    # Weighted sum
output = activation(z)               # Non-linearity
```
**In your work:** Every neuron in BERT does this!

---

### **2. Forward Propagation**
```
Input → Layer1 → Layer2 → Layer3 → Output
```
**What happens:** Data flows forward through weights and activations to make predictions.

**In your work:** `embedding_model.encode("text")` does forward prop through 12 layers.

---

### **3. Backpropagation**
```
Error at output → flows backward → calculates ∂Loss/∂Weight for each weight
```
**What happens:** Chain rule computes how much each weight contributed to error.

**In your work:** How BERT learned from 3.3B words.

---

### **4. Gradient Descent**
```python
weight = weight - (learning_rate × gradient)  # Move downhill toward lower error
```
**Hyperparameters:**
- LR too high → Explodes
- LR too low → Slow convergence
- Sweet spot → Steady progress

**In your work:** BERT fine-tuning uses LR = 2e-5 (tiny because weights already good).

---

### **5. Activation Functions**

| Function | Formula | Use |
|----------|---------|-----|
| **ReLU** | `max(0, z)` | Hidden layers (fast, prevents vanishing gradients) |
| **Sigmoid** | `1/(1+e^(-z))` | Binary output (0-1 probabilities) |
| **Softmax** | `e^zi / Σe^zj` | Multi-class output (probabilities sum to 1) |

**Why needed:** Without them, layers = just linear algebra (no learning complex patterns).

---

### **6. The Training Loop**
```python
for epoch in range(epochs):
    predictions = forward(X)           # 1. Make predictions
    loss = compute_loss(y, predictions) # 2. Measure error
    gradients = backward(X, y)          # 3. Calculate gradients
    update_weights(gradients, lr)       # 4. Update weights
```
**1 Epoch** = Seeing entire dataset once

---

## 🔑 Hidden Concepts

### **Decision Boundaries**
- **Single neuron:** Straight line only (linear) → ~85% accuracy
- **Multi-layer:** Curved boundaries (non-linear) → ~95% accuracy

**Why:** `f(x) = σ(W3*σ(W2*σ(W1*x)))` creates non-linearity!

---

### **Cache Pattern (Why Block 8 needs it)**
```python
# Forward: Store intermediate values
forward():
    a1 = relu(W1*x)
    a2 = relu(W2*a1)
    cache = {'a1': a1, 'a2': a2}  # ← Save for backprop!

# Backward: Use cached values
backward():
    dW2 = ... using a1  # ← Need from forward pass
    dW1 = ... using x
```

---

### **One-Hot Encoding**
```python
Label: 3  →  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                    ↑ position 3
```
**Why:** Networks output probability vectors, not integers.

---

### **Overfitting vs Underfitting**
```
Underfitting: Train=60%, Test=58%  → Model too simple
Good Fit:     Train=95%, Test=92%  → Perfect!
Overfitting:  Train=99%, Test=75%  → Memorized training data
```
**Fix overfitting:** Dropout, L2 regularization, more data, early stopping

---

## 🔗 Connection to Your Work

### **Your RAG Pipeline (What's Actually Happening)**
```python
embedding = model.encode("contract clause")

# Behind the scenes:
1. Tokenize: "contract clause" → [101, 3820, 9897, 102]
2. Embed: tokens → 768-dim vectors
3. Forward through 12 layers (what you built × 12!)
4. Pool: average token embeddings → final 768-dim vector
5. Store in Qdrant
```

---

### **Fine-Tuning BERT**
```python
trainer = Trainer(
    model=bert,
    args=TrainingArguments(
        learning_rate=2e-5,      # Tiny (weights already good)
        num_train_epochs=3,       # Few (prevent overfitting)
        batch_size=16
    )
)
trainer.train()  # ← Uses YOUR training loop (forward→loss→backward→update)!
```

**What's being updated:** 110M parameters using backprop + gradient descent!

---

### **Debugging Model Issues**

| Problem | Check | Fix |
|---------|-------|-----|
| **Low accuracy** | Loss curve decreasing? | More epochs, better LR |
| **Overfitting** | Train acc >> Test acc? | Dropout, regularization, more data |
| **Exploding loss** | Loss = NaN or inf? | Lower learning rate, gradient clipping |
| **Slow learning** | Loss barely decreasing? | Higher LR, check data normalization |

---

## 📊 Block Summary

| Block | What You Built | Key Takeaway |
|-------|---------------|--------------|
| **1-2** | Single neuron | Basic unit: weighted sum + activation |
| **3-4** | Gradient descent | Learning = update weights based on error |
| **5-6** | Multi-layer network | Depth enables non-linear patterns |
| **7-9** | Digit classifier (17K params) | Real problem, 95%+ accuracy |
| **10** | BERT connection | Same principles, just 12 layers + attention |
| **11** | Production ML | Overfitting, regularization, monitoring |

---

## 🎓 Key Equations
```python
# Forward pass (one layer)
z = W·x + b
a = activation(z)

# Backpropagation (chain rule)
dW = (1/m) · X^T · (a - y)
db = (1/m) · sum(a - y)

# Weight update (gradient descent)
W = W - lr · dW
b = b - lr · db

# Loss (binary cross-entropy)
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

# Loss (categorical cross-entropy)  
L = -Σ(y_true · log(y_pred))
```

---

## 🚀 Next Steps

### **For your enterprise RAG Work:**
1. ✅ Understand embedding model internals (you now do!)
2. ✅ Fine-tune BERT confidently (same concepts!)
3. ✅ Debug RAG pipeline (check embeddings, loss curves)
4. ✅ Optimize hyperparameters (LR, batch size, epochs)

### **Further Learning:**
- Attention mechanisms (key to transformers)
- Transformer architecture deep dive
- Advanced RAG techniques
- Model deployment & serving

---

## 💡 Most Important Insights

1. **All neural networks use the same core loop:** forward → loss → backward → update
2. **BERT/GPT = Your network × bigger:** Same backprop, just 12+ layers and billions of params
3. **Embeddings = layer outputs:** The 768-dim vectors are literally outputs from layer 12
4. **Fine-tuning = transfer learning:** Start with good weights, make small adjustments
5. **Production debugging = same principles:** Check loss, accuracy, overfitting, learning rate

**You now understand what happens inside every AI model you use!** 🎉