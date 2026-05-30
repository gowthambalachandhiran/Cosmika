"""
QuantumSimulator
================
A Python quantum computing simulation library supporting 1–8 qubits.

Public API
----------
Core classes:
  QubitState          — complex state-vector for n-qubit systems
  GateSimulator       — stateful, fluent gate-application engine
  CircuitSimulator    — declarative circuit builder with run/draw/export

Quick-start
-----------
>>> from quantum_simulator import GateSimulator, CircuitSimulator

>>> # Bell state via GateSimulator (imperative API)
>>> sim = GateSimulator(2)
>>> sim.h(0).cnot(0, 1)
>>> print(sim.state.dirac_notation())

>>> # Grover's search on 4 qubits
>>> grover = CircuitSimulator.grover(4, marked=[5, 11])
>>> print(grover.run(shots=2048))
"""

__version__ = "1.0.0"
__author__  = "QuantumSimulator Project"

from .qubit             import QubitState, QuantumStateError, MAX_QUBITS, MIN_QUBITS
from .gate_simulator    import GateSimulator, GateSimulatorError
from .circuit_simulator import CircuitSimulator, Instruction
from .visualization     import (
    state_table,
    histogram_text,
    plot_probabilities,
    plot_histogram,
    plot_bloch_sphere,
)
from . import gates

__all__ = [
    "QubitState", "QuantumStateError", "MAX_QUBITS", "MIN_QUBITS",
    "GateSimulator", "GateSimulatorError",
    "CircuitSimulator", "Instruction",
    "gates",
    "state_table", "histogram_text",
    "plot_probabilities", "plot_histogram", "plot_bloch_sphere",
]
