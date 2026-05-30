"""
QuantumSimulator.gate_simulator
================================
GateSimulator — the engine that applies quantum gates to QubitState objects.

Architecture
------------
GateSimulator owns a QubitState and exposes a fluent gate-application API.
Every gate method mutates the internal state and returns `self`, enabling
method chaining:

    sim = GateSimulator(3)
    sim.h(0).cnot(0, 1).cnot(0, 2).measure()

Gate application uses the "full-register lift" technique:
  1. For single-qubit gates on qubit k of an n-qubit register,
     the 2^n × 2^n unitary is  I ⊗ ... ⊗ G_k ⊗ ... ⊗ I
     (G on position k, I everywhere else).
  2. For multi-qubit gates the targets are permuted to adjacent positions,
     the gate applied, then permuted back — no ancilla needed.
"""

from __future__ import annotations
import numpy as np
from typing import Callable, List, Optional

from .qubit import QubitState, QuantumStateError, MAX_QUBITS
from . import gates as G


class GateSimulatorError(Exception):
    """Raised for misuse of GateSimulator."""


class GateSimulator:
    """
    Stateful quantum gate simulator for 1–8 qubits.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (1–8).
    initial_state : int or np.ndarray, optional
        Computational-basis index (default 0 = |0...0⟩) or
        a pre-normalised amplitude vector.

    Examples
    --------
    Bell state:
    >>> sim = GateSimulator(2)
    >>> sim.h(0).cnot(0, 1)
    >>> print(sim.state.dirac_notation())
    (0.707+0.000j)|00⟩ + (0.707+0.000j)|11⟩

    3-qubit GHZ state:
    >>> sim = GateSimulator(3).h(0).cnot(0,1).cnot(0,2)
    >>> print(sim.state.dirac_notation())
    (0.707+0.000j)|000⟩ + (0.707+0.000j)|111⟩
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        n_qubits: int,
        initial_state: int | np.ndarray = 0,
    ) -> None:
        self._state = QubitState(n_qubits, initial_state)
        self._history: list[str] = []       # gate application log
        self._snapshot: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    #  State access                                                        #
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> QubitState:
        """The current QubitState (read-only view)."""
        return self._state

    @property
    def n_qubits(self) -> int:
        return self._state.n_qubits

    @property
    def history(self) -> list[str]:
        """Ordered log of gate operations applied."""
        return list(self._history)

    def snapshot(self) -> "GateSimulator":
        """Save current amplitude vector for later restore."""
        self._snapshot = self._state.amplitudes
        return self

    def restore(self) -> "GateSimulator":
        """Restore to last snapshot."""
        if self._snapshot is None:
            raise GateSimulatorError("No snapshot saved — call snapshot() first.")
        self._state = QubitState(self.n_qubits, self._snapshot)
        self._history.append("RESTORE")
        return self

    def reset(self) -> "GateSimulator":
        """Reset all qubits to |0...0⟩."""
        self._state = QubitState(self.n_qubits, 0)
        self._history.append("RESET")
        return self

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _validate_qubit(self, q: int, label: str = "qubit") -> None:
        if not (0 <= q < self.n_qubits):
            raise GateSimulatorError(
                f"{label} index {q} out of range for {self.n_qubits}-qubit register."
            )

    def _apply_single(self, gate_matrix: np.ndarray, qubit: int) -> None:
        """Lift a 2×2 gate to the full 2^n register and apply."""
        n = self.n_qubits
        # Build I ⊗ ... ⊗ G ⊗ ... ⊗ I  (G at position qubit, big-endian)
        full = None
        for k in range(n):
            m = gate_matrix if k == qubit else np.eye(2, dtype=np.complex128)
            full = m if full is None else np.kron(full, m)
        new_vec = full @ self._state.amplitudes
        self._state._set_state(new_vec)

    def _apply_two_qubit(
        self, gate_matrix: np.ndarray, control: int, target: int
    ) -> None:
        """
        Apply a 4×4 two-qubit gate to (control, target) qubits.
        Uses state-vector re-indexing (no tensor-product lift needed for pairs).
        """
        n = self.n_qubits
        dim = 2**n
        vec = self._state.amplitudes
        new_vec = np.zeros(dim, dtype=np.complex128)

        # Iterate over all basis states, group by the 2-qubit subspace
        visited = set()
        for i in range(dim):
            c_bit = (i >> (n - 1 - control)) & 1
            t_bit = (i >> (n - 1 - target)) & 1
            sub_idx = (c_bit << 1) | t_bit

            # The four basis states that share the same "rest" bits
            rest_mask = ~(
                (1 << (n - 1 - control)) | (1 << (n - 1 - target))
            ) & (dim - 1)
            rest = i & rest_mask
            if rest in visited:
                continue
            visited.add(rest)

            indices = []
            for sub in range(4):
                cb = (sub >> 1) & 1
                tb = sub & 1
                idx = rest | (cb << (n - 1 - control)) | (tb << (n - 1 - target))
                indices.append(idx)

            sub_vec = np.array([vec[j] for j in indices], dtype=np.complex128)
            out_sub = gate_matrix @ sub_vec
            for j, val in zip(indices, out_sub):
                new_vec[j] = val

        self._state._set_state(new_vec)

    def _apply_three_qubit(
        self, gate_matrix: np.ndarray, q0: int, q1: int, q2: int
    ) -> None:
        """Apply an 8×8 three-qubit gate to (q0, q1, q2)."""
        n = self.n_qubits
        dim = 2**n
        vec = self._state.amplitudes
        new_vec = np.zeros(dim, dtype=np.complex128)

        visited = set()
        for i in range(dim):
            b0 = (i >> (n - 1 - q0)) & 1
            b1 = (i >> (n - 1 - q1)) & 1
            b2 = (i >> (n - 1 - q2)) & 1

            rest_mask = ~(
                (1 << (n-1-q0)) | (1 << (n-1-q1)) | (1 << (n-1-q2))
            ) & (dim - 1)
            rest = i & rest_mask
            if rest in visited:
                continue
            visited.add(rest)

            indices = []
            for sub in range(8):
                b_0 = (sub >> 2) & 1
                b_1 = (sub >> 1) & 1
                b_2 = sub & 1
                idx = (rest
                       | (b_0 << (n-1-q0))
                       | (b_1 << (n-1-q1))
                       | (b_2 << (n-1-q2)))
                indices.append(idx)

            sub_vec = np.array([vec[j] for j in indices], dtype=np.complex128)
            out_sub = gate_matrix @ sub_vec
            for j, val in zip(indices, out_sub):
                new_vec[j] = val

        self._state._set_state(new_vec)

    def _log(self, op: str) -> None:
        self._history.append(op)

    # ------------------------------------------------------------------ #
    #  Single-qubit gates                                                  #
    # ------------------------------------------------------------------ #

    def i(self, qubit: int) -> "GateSimulator":
        """Apply Identity gate (no-op, useful for circuit notation)."""
        self._validate_qubit(qubit)
        self._log(f"I({qubit})")
        return self

    def x(self, qubit: int) -> "GateSimulator":
        """Apply Pauli-X (NOT) gate."""
        self._validate_qubit(qubit)
        self._apply_single(G.pauli_x(), qubit)
        self._log(f"X({qubit})")
        return self

    def y(self, qubit: int) -> "GateSimulator":
        """Apply Pauli-Y gate."""
        self._validate_qubit(qubit)
        self._apply_single(G.pauli_y(), qubit)
        self._log(f"Y({qubit})")
        return self

    def z(self, qubit: int) -> "GateSimulator":
        """Apply Pauli-Z gate."""
        self._validate_qubit(qubit)
        self._apply_single(G.pauli_z(), qubit)
        self._log(f"Z({qubit})")
        return self

    def h(self, qubit: int) -> "GateSimulator":
        """Apply Hadamard gate."""
        self._validate_qubit(qubit)
        self._apply_single(G.hadamard(), qubit)
        self._log(f"H({qubit})")
        return self

    def s(self, qubit: int) -> "GateSimulator":
        """Apply S (Phase) gate."""
        self._validate_qubit(qubit)
        self._apply_single(G.phase_s(), qubit)
        self._log(f"S({qubit})")
        return self

    def sdg(self, qubit: int) -> "GateSimulator":
        """Apply S† gate."""
        self._validate_qubit(qubit)
        self._apply_single(G.phase_s_dagger(), qubit)
        self._log(f"Sdg({qubit})")
        return self

    def t(self, qubit: int) -> "GateSimulator":
        """Apply T gate."""
        self._validate_qubit(qubit)
        self._apply_single(G.t_gate(), qubit)
        self._log(f"T({qubit})")
        return self

    def tdg(self, qubit: int) -> "GateSimulator":
        """Apply T† gate."""
        self._validate_qubit(qubit)
        self._apply_single(G.t_dagger(), qubit)
        self._log(f"Tdg({qubit})")
        return self

    def rx(self, qubit: int, theta: float) -> "GateSimulator":
        """Apply Rx(θ) rotation gate."""
        self._validate_qubit(qubit)
        self._apply_single(G.rx(theta), qubit)
        self._log(f"Rx({qubit}, θ={theta:.4f})")
        return self

    def ry(self, qubit: int, theta: float) -> "GateSimulator":
        """Apply Ry(θ) rotation gate."""
        self._validate_qubit(qubit)
        self._apply_single(G.ry(theta), qubit)
        self._log(f"Ry({qubit}, θ={theta:.4f})")
        return self

    def rz(self, qubit: int, theta: float) -> "GateSimulator":
        """Apply Rz(θ) rotation gate."""
        self._validate_qubit(qubit)
        self._apply_single(G.rz(theta), qubit)
        self._log(f"Rz({qubit}, θ={theta:.4f})")
        return self

    def r1(self, qubit: int, lam: float) -> "GateSimulator":
        """Apply R1(λ) phase gate."""
        self._validate_qubit(qubit)
        self._apply_single(G.r1(lam), qubit)
        self._log(f"R1({qubit}, λ={lam:.4f})")
        return self

    def u3(self, qubit: int, theta: float, phi: float, lam: float) -> "GateSimulator":
        """Apply general U3(θ, φ, λ) gate."""
        self._validate_qubit(qubit)
        self._apply_single(G.u3(theta, phi, lam), qubit)
        self._log(f"U3({qubit}, θ={theta:.4f}, φ={phi:.4f}, λ={lam:.4f})")
        return self

    def custom_gate(self, qubit: int, matrix: np.ndarray, name: str = "Custom") -> "GateSimulator":
        """Apply any user-supplied 2×2 unitary gate."""
        self._validate_qubit(qubit)
        matrix = np.asarray(matrix, dtype=np.complex128)
        if matrix.shape != (2, 2):
            raise GateSimulatorError("custom_gate requires a 2×2 matrix.")
        # Check unitarity
        prod = matrix @ matrix.conj().T
        if not np.allclose(prod, np.eye(2), atol=1e-6):
            raise GateSimulatorError("Matrix is not unitary (U†U ≠ I).")
        self._apply_single(matrix, qubit)
        self._log(f"{name}({qubit})")
        return self

    # ------------------------------------------------------------------ #
    #  Two-qubit gates                                                     #
    # ------------------------------------------------------------------ #

    def cnot(self, control: int, target: int) -> "GateSimulator":
        """Apply CNOT (CX) gate."""
        self._validate_qubit(control, "control")
        self._validate_qubit(target, "target")
        if control == target:
            raise GateSimulatorError("control and target must be different qubits.")
        self._apply_two_qubit(G.cnot(), control, target)
        self._log(f"CNOT({control}→{target})")
        return self

    def cx(self, control: int, target: int) -> "GateSimulator":
        """Alias for cnot."""
        return self.cnot(control, target)

    def cz(self, control: int, target: int) -> "GateSimulator":
        """Apply Controlled-Z gate."""
        self._validate_qubit(control, "control")
        self._validate_qubit(target, "target")
        if control == target:
            raise GateSimulatorError("control and target must be different.")
        self._apply_two_qubit(G.cz(), control, target)
        self._log(f"CZ({control},{target})")
        return self

    def swap(self, q0: int, q1: int) -> "GateSimulator":
        """Apply SWAP gate."""
        self._validate_qubit(q0)
        self._validate_qubit(q1)
        if q0 == q1:
            raise GateSimulatorError("SWAP requires two different qubits.")
        self._apply_two_qubit(G.swap(), q0, q1)
        self._log(f"SWAP({q0},{q1})")
        return self

    def iswap(self, q0: int, q1: int) -> "GateSimulator":
        """Apply iSWAP gate."""
        self._validate_qubit(q0)
        self._validate_qubit(q1)
        self._apply_two_qubit(G.iswap(), q0, q1)
        self._log(f"iSWAP({q0},{q1})")
        return self

    def ch(self, control: int, target: int) -> "GateSimulator":
        """Apply Controlled-Hadamard gate."""
        self._validate_qubit(control, "control")
        self._validate_qubit(target, "target")
        self._apply_two_qubit(G.controlled_h(), control, target)
        self._log(f"CH({control}→{target})")
        return self

    def cp(self, control: int, target: int, lam: float) -> "GateSimulator":
        """Apply Controlled-Phase CP(λ) gate."""
        self._validate_qubit(control, "control")
        self._validate_qubit(target, "target")
        self._apply_two_qubit(G.controlled_phase(lam), control, target)
        self._log(f"CP({control},{target}, λ={lam:.4f})")
        return self

    # ------------------------------------------------------------------ #
    #  Three-qubit gates                                                   #
    # ------------------------------------------------------------------ #

    def toffoli(self, c0: int, c1: int, target: int) -> "GateSimulator":
        """Apply Toffoli (CCX) gate — two controls, one target."""
        for label, q in [("c0", c0), ("c1", c1), ("target", target)]:
            self._validate_qubit(q, label)
        if len({c0, c1, target}) != 3:
            raise GateSimulatorError("Toffoli requires 3 distinct qubits.")
        self._apply_three_qubit(G.toffoli(), c0, c1, target)
        self._log(f"CCX({c0},{c1}→{target})")
        return self

    def ccx(self, c0: int, c1: int, target: int) -> "GateSimulator":
        """Alias for toffoli."""
        return self.toffoli(c0, c1, target)

    def fredkin(self, control: int, t0: int, t1: int) -> "GateSimulator":
        """Apply Fredkin (CSWAP) gate — one control, two targets."""
        for label, q in [("control", control), ("t0", t0), ("t1", t1)]:
            self._validate_qubit(q, label)
        if len({control, t0, t1}) != 3:
            raise GateSimulatorError("Fredkin requires 3 distinct qubits.")
        self._apply_three_qubit(G.fredkin(), control, t0, t1)
        self._log(f"CSWAP({control},{t0},{t1})")
        return self

    def cswap(self, control: int, t0: int, t1: int) -> "GateSimulator":
        """Alias for fredkin."""
        return self.fredkin(control, t0, t1)

    # ------------------------------------------------------------------ #
    #  Oracle gates                                                        #
    # ------------------------------------------------------------------ #

    def grover_oracle(self, marked_states: list[int]) -> "GateSimulator":
        """
        Apply Grover phase oracle: flip phase of each marked basis state.

        Parameters
        ----------
        marked_states : list of int
            Indices of states to mark (phase flip).
        """
        oracle_mat = G.grover_oracle(marked_states, self.n_qubits)
        new_vec = oracle_mat @ self._state.amplitudes
        self._state._set_state(new_vec)
        self._log(f"GroverOracle({marked_states})")
        return self

    def phase_oracle(self, f: Callable[[int], int]) -> "GateSimulator":
        """
        Apply phase oracle  Uf|x⟩ = (−1)^{f(x)}|x⟩.

        Parameters
        ----------
        f : callable
            Boolean function {0,...,2^n-1} → {0,1}.
        """
        oracle_mat = G.phase_oracle(f, self.n_qubits)
        new_vec = oracle_mat @ self._state.amplitudes
        self._state._set_state(new_vec)
        self._log("PhaseOracle(f)")
        return self

    def diffusion(self) -> "GateSimulator":
        """Apply Grover diffusion operator  2|s⟩⟨s| - I."""
        diff_mat = G.diffusion_operator(self.n_qubits)
        new_vec = diff_mat @ self._state.amplitudes
        self._state._set_state(new_vec)
        self._log("Diffusion")
        return self

    # ------------------------------------------------------------------ #
    #  Hadamard transform                                                  #
    # ------------------------------------------------------------------ #

    def had_all(self) -> "GateSimulator":
        """Apply Hadamard to every qubit  H^⊗n."""
        for q in range(self.n_qubits):
            self._apply_single(G.hadamard(), q)
        self._log("H_all")
        return self

    # ------------------------------------------------------------------ #
    #  Measurement                                                         #
    # ------------------------------------------------------------------ #

    def measure(
        self,
        shots: int = 1024,
        seed: int | None = None,
    ) -> dict[str, int]:
        """
        Measure all qubits in the computational basis.

        Returns
        -------
        dict mapping bitstring → count
        """
        return self._state.measure(shots=shots, seed=seed)

    def measure_once(self, seed: int | None = None) -> str:
        """Single-shot projective measurement."""
        return self._state.measure_once(seed=seed)

    # ------------------------------------------------------------------ #
    #  Representation                                                      #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"GateSimulator(n={self.n_qubits}, "
            f"state={self._state.dirac_notation()}, "
            f"ops={len(self._history)})"
        )

    def summary(self) -> str:
        """Pretty-print the current state and gate history."""
        lines = [
            "=" * 60,
            f"  GateSimulator  |  {self.n_qubits} qubits",
            "=" * 60,
            f"  State : {self._state.dirac_notation()}",
            f"  Dim   : {self._state.dim}",
            "",
            "  Probabilities:",
        ]
        fmt = f"{{:0{self.n_qubits}b}}"
        for i, p in enumerate(self._state.probabilities):
            if p > 1e-6:
                lines.append(f"    |{fmt.format(i)}⟩  {p:.6f}  ({p*100:.2f}%)")
        lines += [
            "",
            f"  Gate history ({len(self._history)} ops):",
        ]
        for op in self._history:
            lines.append(f"    {op}")
        lines.append("=" * 60)
        return "\n".join(lines)
