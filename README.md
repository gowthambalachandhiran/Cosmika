# 🌌 Cosmika

**Cosmika** is a lightweight Python quantum computing simulation library designed for learning, experimentation, and visualization of quantum algorithms.

Built with simplicity in mind, Cosmika provides an intuitive API for constructing qubit states, applying quantum gates, building quantum circuits, and visualizing quantum state evolution.

---

## ✨ Features

* 🔹 Single and multi-qubit state simulation
* 🔹 Common quantum gates (X, Y, Z, H, S, T, CNOT, SWAP, etc.)
* 🔹 Circuit-based and imperative simulation APIs
* 🔹 State vector visualization
* 🔹 Measurement and probability analysis
* 🔹 Dirac notation support
* 🔹 Educational focus with readable implementations
* 🔹 Pure Python implementation

---

## 📦 Installation

```bash
pip install cosmika
```

Or install from source:

```bash
git clone https://github.com/gowthambalachandhiran/Cosmika.git
cd Cosmika
pip install -e .
```

---

## 🚀 Quick Start

### Working with Qubits

```python
from Cosmika.qubit import QubitState

q = QubitState.zero()
print(q)
```

---

### Using GateSimulator

```python
from Cosmika.gate_simulator import GateSimulator

sim = GateSimulator(1)

sim.h(0)

print(sim.state)
```

---

### Creating a Bell State

```python
from Cosmika.gate_simulator import GateSimulator

sim = GateSimulator(2)

sim.h(0)
sim.cnot(0, 1)

print(sim.state.dirac_notation())
```

Expected output:

```text
(|00⟩ + |11⟩)/√2
```

---

### Circuit-Based API

```python
from Cosmika.gate_simulator import CircuitSimulator

circuit = CircuitSimulator(2)

circuit.h(0)
circuit.cnot(0, 1)

result = circuit.run()

print(result)
```

---

## 📊 Visualization

### State Table

```python
from Cosmika.gate_simulator import state_table

state_table(sim.state)
```

### Histogram

```python
from Cosmika.gate_simulator import histogram_text

histogram_text(sim.measure(shots=1024))
```

---

## 📚 Supported Gates

| Gate | Description    |
| ---- | -------------- |
| X    | Pauli-X (NOT)  |
| Y    | Pauli-Y        |
| Z    | Pauli-Z        |
| H    | Hadamard       |
| S    | Phase Gate     |
| T    | π/8 Gate       |
| CNOT | Controlled NOT |
| CZ   | Controlled Z   |
| SWAP | Swap Gate      |

---

## 🏗 Package Structure

```text
Cosmika/
│
├── Cosmika/
│   ├── __init__.py
│   ├── qubit.py
│   ├── gates.py
│   │
│   └── gate_simulator/
│       ├── gate_simulator.py
│       ├── circuit_simulator.py
│       ├── visualization.py
│       └── __init__.py
│
├── tests/
├── examples/
└── README.md
```

---

## 🎯 Roadmap

Future enhancements planned for Cosmika:

* Bloch Sphere Visualization
* Quantum Fourier Transform (QFT)
* Grover's Search Algorithm
* Deutsch-Jozsa Algorithm
* Quantum Teleportation
* Noise Models
* Density Matrix Simulation
* OpenQASM Import/Export
* Interactive Visualizations

---

## 🤝 Contributing

Contributions are welcome.

Feel free to open issues, submit pull requests, or suggest new features.

```bash
git checkout -b feature/my-feature
git commit -m "Add awesome feature"
git push origin feature/my-feature
```

---

## 📜 License

MIT License

---

## 👨‍💻 Author

**Gowtham Balachandhiran**

Passionate about Quantum Computing, Artificial Intelligence, and building educational tools that make advanced concepts accessible to everyone.

GitHub: https://github.com/gowthambalachandhiran
