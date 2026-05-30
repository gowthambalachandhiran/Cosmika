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

from .gate_simulator import (
    GateSimulator,
    CircuitSimulator,
    state_table,
    histogram_text
)

from .gates import *
from .qubit import *

__version__ = "0.1.0"