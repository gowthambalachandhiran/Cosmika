"""
QuantumSimulator.gates
======================
Gate matrix library — all standard single-qubit and multi-qubit gates.

Every gate is returned as a complex128 NumPy ndarray.

Implemented gates
-----------------
Single-qubit (2×2)
  Identity   I
  Pauli-X    X   (NOT gate)
  Pauli-Y    Y
  Pauli-Z    Z
  Hadamard   H
  Phase S    S   (√Z)
  T gate     T   (√S = ⁴√Z,  π/8 gate)
  Rx(θ)      rotation about X-axis by angle θ
  Ry(θ)      rotation about Y-axis by angle θ
  Rz(θ)      rotation about Z-axis by angle θ
  R1(λ)      phase rotation  [[1,0],[0,e^iλ]]
  U3(θ,φ,λ)  general single-qubit SU(2) gate

Two-qubit (4×4)
  CNOT       Controlled-X
  CZ         Controlled-Z
  SWAP       SWAP
  iSWAP      iSWAP
  CH         Controlled-H

Three-qubit (8×8)
  Toffoli    CCX (Controlled-Controlled-X)
  Fredkin    CSWAP (Controlled-SWAP)

Oracle gates
  GroverOracle(marked_states, n_qubits)   phase oracle Uω
  PhaseOracle(f, n_qubits)                general phase oracle for boolean f
"""

from __future__ import annotations
import numpy as np
from typing import Callable

# Convenience alias
_c = np.complex128
_I = np.eye(2, dtype=_c)
_SQRT2_INV = 1.0 / np.sqrt(2)


# =========================================================================== #
#  Single-qubit gates                                                          #
# =========================================================================== #

def identity() -> np.ndarray:
    """Identity gate  I = [[1,0],[0,1]]"""
    return np.eye(2, dtype=_c)


def pauli_x() -> np.ndarray:
    """Pauli-X (NOT / bit-flip) gate."""
    return np.array([[0, 1], [1, 0]], dtype=_c)


def pauli_y() -> np.ndarray:
    """Pauli-Y gate."""
    return np.array([[0, -1j], [1j, 0]], dtype=_c)


def pauli_z() -> np.ndarray:
    """Pauli-Z (phase-flip) gate."""
    return np.array([[1, 0], [0, -1]], dtype=_c)


def hadamard() -> np.ndarray:
    """Hadamard gate  H = (X+Z)/√2."""
    return np.array(
        [[_SQRT2_INV, _SQRT2_INV],
         [_SQRT2_INV, -_SQRT2_INV]],
        dtype=_c,
    )


def phase_s() -> np.ndarray:
    """Phase / S gate  = √Z = [[1,0],[0,i]]."""
    return np.array([[1, 0], [0, 1j]], dtype=_c)


def phase_s_dagger() -> np.ndarray:
    """S† (S-dagger) gate — conjugate transpose of S."""
    return np.array([[1, 0], [0, -1j]], dtype=_c)


def t_gate() -> np.ndarray:
    """T gate (π/8 gate) = [[1,0],[0,e^{iπ/4}]] = ⁴√Z."""
    return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=_c)


def t_dagger() -> np.ndarray:
    """T† gate — conjugate transpose of T."""
    return np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=_c)


def rx(theta: float) -> np.ndarray:
    """Rotation about X-axis by angle θ:  Rx(θ) = e^{-iθX/2}."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=_c)


def ry(theta: float) -> np.ndarray:
    """Rotation about Y-axis by angle θ:  Ry(θ) = e^{-iθY/2}."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=_c)


def rz(theta: float) -> np.ndarray:
    """Rotation about Z-axis by angle θ:  Rz(θ) = e^{-iθZ/2}."""
    return np.array(
        [[np.exp(-1j * theta / 2), 0],
         [0, np.exp(1j * theta / 2)]],
        dtype=_c,
    )


def r1(lam: float) -> np.ndarray:
    """Phase rotation  R1(λ) = [[1,0],[0,e^{iλ}]]."""
    return np.array([[1, 0], [0, np.exp(1j * lam)]], dtype=_c)


def u3(theta: float, phi: float, lam: float) -> np.ndarray:
    """
    General single-qubit gate U3(θ, φ, λ):
      [[cos(θ/2),          -e^{iλ}sin(θ/2)],
       [e^{iφ}sin(θ/2),    e^{i(φ+λ)}cos(θ/2)]]
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array(
        [[c, -np.exp(1j * lam) * s],
         [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]],
        dtype=_c,
    )


# =========================================================================== #
#  Two-qubit gates (4×4)                                                       #
# =========================================================================== #

def cnot() -> np.ndarray:
    """CNOT (Controlled-X) gate — qubit 0 is control, qubit 1 is target."""
    return np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]],
        dtype=_c,
    )


def cz() -> np.ndarray:
    """Controlled-Z gate."""
    return np.diag([1, 1, 1, -1]).astype(_c)


def swap() -> np.ndarray:
    """SWAP gate."""
    return np.array(
        [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]],
        dtype=_c,
    )


def iswap() -> np.ndarray:
    """iSWAP gate."""
    return np.array(
        [[1, 0, 0, 0],
         [0, 0, 1j, 0],
         [0, 1j, 0, 0],
         [0, 0, 0, 1]],
        dtype=_c,
    )


def controlled_h() -> np.ndarray:
    """Controlled-Hadamard gate."""
    h = hadamard()
    mat = np.eye(4, dtype=_c)
    mat[2:, 2:] = h
    return mat


def controlled_phase(lam: float) -> np.ndarray:
    """Controlled-Phase gate  CP(λ)."""
    return np.diag([1, 1, 1, np.exp(1j * lam)]).astype(_c)


# =========================================================================== #
#  Three-qubit gates (8×8)                                                     #
# =========================================================================== #

def toffoli() -> np.ndarray:
    """
    Toffoli / CCX gate (Controlled-Controlled-NOT).
    Qubit ordering: control0, control1, target.
    """
    mat = np.eye(8, dtype=_c)
    mat[6, 6] = 0; mat[6, 7] = 1
    mat[7, 7] = 0; mat[7, 6] = 1
    return mat


def fredkin() -> np.ndarray:
    """
    Fredkin / CSWAP gate (Controlled-SWAP).
    Qubit ordering: control, target0, target1.
    """
    mat = np.eye(8, dtype=_c)
    mat[5, 5] = 0; mat[5, 6] = 1
    mat[6, 6] = 0; mat[6, 5] = 1
    return mat


# =========================================================================== #
#  Oracle gates                                                                #
# =========================================================================== #

def grover_oracle(marked_states: list[int], n_qubits: int) -> np.ndarray:
    """
    Phase oracle  Uω|x⟩ = -|x⟩  if x in marked_states, else |x⟩.

    Parameters
    ----------
    marked_states : list of int
        Basis state indices to mark (flip phase).
    n_qubits : int
        Total number of qubits (1–8).

    Returns
    -------
    ndarray of shape (2^n, 2^n)
    """
    dim = 2**n_qubits
    mat = np.eye(dim, dtype=_c)
    for s in marked_states:
        if not (0 <= s < dim):
            raise ValueError(f"State {s} out of range for {n_qubits} qubits.")
        mat[s, s] = -1.0
    return mat


def phase_oracle(f: Callable[[int], int], n_qubits: int) -> np.ndarray:
    """
    General phase oracle  Uf|x⟩ = (-1)^{f(x)}|x⟩.

    Parameters
    ----------
    f : callable
        Boolean function f: {0,...,2^n-1} → {0,1}.
    n_qubits : int
        Number of input qubits.

    Returns
    -------
    ndarray of shape (2^n, 2^n)
    """
    dim = 2**n_qubits
    diag = np.array([(-1.0) ** f(x) for x in range(dim)], dtype=_c)
    return np.diag(diag)


def diffusion_operator(n_qubits: int) -> np.ndarray:
    """
    Grover diffusion operator  2|s⟩⟨s| - I  where |s⟩ = H^⊗n|0⟩.

    Parameters
    ----------
    n_qubits : int

    Returns
    -------
    ndarray of shape (2^n, 2^n)
    """
    dim = 2**n_qubits
    s = np.ones(dim, dtype=_c) / np.sqrt(dim)
    return 2 * np.outer(s, s.conj()) - np.eye(dim, dtype=_c)


# =========================================================================== #
#  Gate catalogue                                                              #
# =========================================================================== #

SINGLE_QUBIT_GATES: dict[str, np.ndarray] = {
    "I":  identity(),
    "X":  pauli_x(),
    "Y":  pauli_y(),
    "Z":  pauli_z(),
    "H":  hadamard(),
    "S":  phase_s(),
    "Sd": phase_s_dagger(),
    "T":  t_gate(),
    "Td": t_dagger(),
}

GATE_DESCRIPTIONS: dict[str, str] = {
    "I":   "Identity",
    "X":   "Pauli-X (NOT / bit-flip)",
    "Y":   "Pauli-Y",
    "Z":   "Pauli-Z (phase-flip)",
    "H":   "Hadamard",
    "S":   "Phase (S = √Z)",
    "Sd":  "S-dagger (S†)",
    "T":   "T gate (π/8, ⁴√Z)",
    "Td":  "T-dagger (T†)",
    "Rx":  "Rotation about X-axis (parameterised θ)",
    "Ry":  "Rotation about Y-axis (parameterised θ)",
    "Rz":  "Rotation about Z-axis (parameterised θ)",
    "R1":  "Phase rotation R1(λ)",
    "U3":  "General single-qubit U3(θ,φ,λ)",
    "CNOT":"Controlled-NOT (2-qubit)",
    "CZ":  "Controlled-Z (2-qubit)",
    "SWAP":"SWAP (2-qubit)",
    "iSWAP":"iSWAP (2-qubit)",
    "CH":  "Controlled-Hadamard (2-qubit)",
    "CP":  "Controlled-Phase CP(λ) (2-qubit)",
    "CCX": "Toffoli / CCX (3-qubit)",
    "CSWAP":"Fredkin / CSWAP (3-qubit)",
    "GroverOracle": "Grover phase oracle (marks states with −1 phase)",
    "PhaseOracle":  "General phase oracle for boolean function f",
    "Diffusion":    "Grover diffusion operator",
}
