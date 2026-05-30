"""
examples/demo.py
================
QuantumSimulator library — interactive demonstration.

Run:  python examples/demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from quantum_simulator import GateSimulator, CircuitSimulator, state_table, histogram_text


def section(title: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


# ─────────────────────────────────────────────────────────────
#  1. Single-qubit gates
# ─────────────────────────────────────────────────────────────
section("1. Single-qubit gate basics")

sim = GateSimulator(1)
print(f"Initial state   : {sim.state.dirac_notation()}")
sim.h(0)
print(f"After H(0)      : {sim.state.dirac_notation()}")
sim.t(0)
print(f"After T(0)      : {sim.state.dirac_notation()}")
sim.h(0)
print(f"After H(0)      : {sim.state.dirac_notation()}")
print(state_table(sim.state))


# ─────────────────────────────────────────────────────────────
#  2. Bell state (2-qubit entanglement)
# ─────────────────────────────────────────────────────────────
section("2. Bell State  |Φ+⟩ = (|00⟩ + |11⟩)/√2")

sim = GateSimulator(2)
sim.h(0).cnot(0, 1)
print(sim.summary())
counts = sim.measure(shots=4096, seed=42)
print(histogram_text(counts))


# ─────────────────────────────────────────────────────────────
#  3. GHZ state (3-qubit)
# ─────────────────────────────────────────────────────────────
section("3. GHZ State  |GHZ⟩ = (|000⟩ + |111⟩)/√2")

sim = GateSimulator(3)
sim.h(0).cnot(0, 1).cnot(0, 2)
print(sim.state.dirac_notation())
counts = sim.measure(shots=2048, seed=0)
print(histogram_text(counts))


# ─────────────────────────────────────────────────────────────
#  4. Toffoli gate
# ─────────────────────────────────────────────────────────────
section("4. Toffoli (CCX) gate  |110⟩ → |111⟩")

sim = GateSimulator(3, 6)  # |110⟩
print(f"Before Toffoli  : {sim.state.dirac_notation()}")
sim.toffoli(0, 1, 2)
print(f"After Toffoli   : {sim.state.dirac_notation()}")


# ─────────────────────────────────────────────────────────────
#  5. Rotation gates (Rx, Ry, Rz)
# ─────────────────────────────────────────────────────────────
section("5. Rotation gates — Ry(π/3)")

sim = GateSimulator(1)
sim.ry(0, np.pi / 3)
print(f"Ry(π/3)|0⟩  : {sim.state.dirac_notation()}")
print(state_table(sim.state))


# ─────────────────────────────────────────────────────────────
#  6. Oracle gates
# ─────────────────────────────────────────────────────────────
section("6. Phase oracle + Grover diffusion (1 step)")

sim = GateSimulator(3)
sim.had_all()
sim.grover_oracle([5])    # Mark |101⟩
sim.diffusion()
print("After 1 Grover iteration (marked state = |101⟩):")
print(state_table(sim.state))


# ─────────────────────────────────────────────────────────────
#  7. CircuitSimulator — built-in templates
# ─────────────────────────────────────────────────────────────
section("7. CircuitSimulator — Bell state template")

circ = CircuitSimulator.bell_state()
circ.draw()
counts = circ.run(shots=2048, seed=1)
print(histogram_text(counts))


# ─────────────────────────────────────────────────────────────
#  8. Grover's search — 4 qubits
# ─────────────────────────────────────────────────────────────
section("8. Grover's Search Algorithm — 4 qubits, marked=[7]")

grover = CircuitSimulator.grover(4, marked=[7], iterations=4)
grover.draw()
counts = grover.run(shots=1024, seed=0)
print(histogram_text(counts))
most_likely = max(counts, key=counts.get)
print(f"\n✓ Most likely measurement: |{most_likely}⟩  (decimal {int(most_likely, 2)})")


# ─────────────────────────────────────────────────────────────
#  9. Quantum Fourier Transform — 3 qubits
# ─────────────────────────────────────────────────────────────
section("9. Quantum Fourier Transform (3 qubits)")

qft = CircuitSimulator.qft(3)
qft.draw()
sv = qft.statevector()
print("QFT(|000⟩) state:")
print(state_table(sv))


# ─────────────────────────────────────────────────────────────
#  10. Circuit inversion
# ─────────────────────────────────────────────────────────────
section("10. Circuit inversion — Bell then Bell†")

bell   = CircuitSimulator.bell_state()
inv    = bell.inverse()
combined = bell.clone().append(inv)
sv = combined.statevector()
print(f"Bell then Bell†: {sv.dirac_notation()}")
print("✓ Returns to |00⟩ (as expected)")


# ─────────────────────────────────────────────────────────────
#  11. Deutsch–Jozsa algorithm
# ─────────────────────────────────────────────────────────────
section("11. Deutsch–Jozsa  (balanced oracle: f(x) = x mod 2)")

n = 3
sim = GateSimulator(n)
sim.had_all()
sim.phase_oracle(lambda x: x % 2)
sim.had_all()
counts = sim.measure(shots=512, seed=0)
print("Balanced oracle — non-zero measurement expected (not |000⟩):")
print(histogram_text(counts))


# ─────────────────────────────────────────────────────────────
#  12. 8-qubit uniform superposition
# ─────────────────────────────────────────────────────────────
section("12. 8-qubit uniform superposition (H^⊗8)")

sim = GateSimulator(8)
sim.had_all()
probs = sim.state.probabilities
print(f"n_qubits : 8  →  dim = {sim.state.dim}")
print(f"Each basis state probability: {probs[0]:.6f}  (=1/256 = {1/256:.6f})")
print(f"Probabilities sum to: {probs.sum():.8f}")
print("✓ Perfect uniform superposition over 256 basis states")


print(f"\n{'═'*60}")
print("  All demos completed successfully!")
print(f"{'═'*60}\n")
