"""
QuantumSimulator.qubit
======================
Core qubit / multi-qubit state-vector representation.

Supports 1–8 qubits (2–256 dimensional complex Hilbert space).
"""

from __future__ import annotations
import numpy as np
from typing import Union


# --------------------------------------------------------------------------- #
#  Constants                                                                   #
# --------------------------------------------------------------------------- #

MAX_QUBITS = 8
MIN_QUBITS = 1
_SQRT2_INV = 1.0 / np.sqrt(2)


class QuantumStateError(Exception):
    """Raised when a qubit state is invalid."""


# --------------------------------------------------------------------------- #
#  QubitState                                                                  #
# --------------------------------------------------------------------------- #

class QubitState:
    """
    Immutable-like state vector for an n-qubit system (1 ≤ n ≤ 8).

    The state is stored as a complex128 NumPy array of length 2**n,
    indexed in big-endian (most-significant qubit first) convention —
    the same convention used by Qiskit's statevector.

    Examples
    --------
    >>> q = QubitState(2)          # |00⟩
    >>> q.amplitudes
    array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
    """

    __slots__ = ("_n", "_state")

    def __init__(
        self,
        n_qubits: int,
        initial_state: Union[int, np.ndarray, None] = 0,
    ) -> None:
        if not (MIN_QUBITS <= n_qubits <= MAX_QUBITS):
            raise QuantumStateError(
                f"n_qubits must be between {MIN_QUBITS} and {MAX_QUBITS}, got {n_qubits}."
            )
        self._n = n_qubits
        dim = 2**n_qubits

        if isinstance(initial_state, int):
            if not (0 <= initial_state < dim):
                raise QuantumStateError(
                    f"Basis state index {initial_state} out of range for {n_qubits} qubits."
                )
            vec = np.zeros(dim, dtype=np.complex128)
            vec[initial_state] = 1.0
            self._state = vec

        elif isinstance(initial_state, np.ndarray):
            vec = np.asarray(initial_state, dtype=np.complex128).ravel()
            if vec.shape != (dim,):
                raise QuantumStateError(
                    f"State vector must have length {dim}, got {vec.shape[0]}."
                )
            norm = np.linalg.norm(vec)
            if not np.isclose(norm, 1.0, atol=1e-6):
                raise QuantumStateError(
                    f"State vector is not normalised (norm = {norm:.6f})."
                )
            self._state = vec.copy()

        else:
            raise QuantumStateError("initial_state must be an int or numpy array.")

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def n_qubits(self) -> int:
        return self._n

    @property
    def dim(self) -> int:
        return 2**self._n

    @property
    def amplitudes(self) -> np.ndarray:
        """Return a copy of the raw complex amplitude vector."""
        return self._state.copy()

    @property
    def probabilities(self) -> np.ndarray:
        """Born-rule probabilities for each basis state."""
        return np.abs(self._state) ** 2

    # ------------------------------------------------------------------ #
    #  Mutation (used by GateSimulator)                                    #
    # ------------------------------------------------------------------ #

    def _set_state(self, new_state: np.ndarray) -> None:
        """Internal — set state vector (normalisation checked)."""
        norm = np.linalg.norm(new_state)
        if not np.isclose(norm, 1.0, atol=1e-4):
            raise QuantumStateError(
                f"Resulting state is not normalised (norm = {norm:.6f}). "
                "Check your gate matrix."
            )
        self._state = new_state.astype(np.complex128)

    # ------------------------------------------------------------------ #
    #  Measurement                                                         #
    # ------------------------------------------------------------------ #

    def measure(self, shots: int = 1024, seed: int | None = None) -> dict[str, int]:
        """
        Simulate projective measurement in the computational basis.

        Parameters
        ----------
        shots : int
            Number of measurement repetitions.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        dict mapping bitstring → count
        """
        rng = np.random.default_rng(seed)
        probs = self.probabilities
        indices = rng.choice(self.dim, size=shots, p=probs)
        fmt = f"{{:0{self._n}b}}"
        counts: dict[str, int] = {}
        for idx in indices:
            key = fmt.format(idx)
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def measure_once(self, seed: int | None = None) -> str:
        """Single-shot measurement; returns bitstring."""
        result = self.measure(shots=1, seed=seed)
        return next(iter(result))

    # ------------------------------------------------------------------ #
    #  Collapse                                                            #
    # ------------------------------------------------------------------ #

    def collapse(self, qubit_index: int, outcome: int) -> "QubitState":
        """
        Post-measurement collapse of qubit `qubit_index` to `outcome` (0 or 1).
        Returns a new QubitState.
        """
        if not (0 <= qubit_index < self._n):
            raise QuantumStateError(f"qubit_index {qubit_index} out of range.")
        if outcome not in (0, 1):
            raise QuantumStateError("outcome must be 0 or 1.")

        new_state = self._state.copy()
        for i in range(self.dim):
            bit = (i >> (self._n - 1 - qubit_index)) & 1
            if bit != outcome:
                new_state[i] = 0.0

        norm = np.linalg.norm(new_state)
        if norm < 1e-10:
            raise QuantumStateError(
                f"Collapse to qubit {qubit_index}={outcome} has zero probability."
            )
        new_state /= norm
        return QubitState(self._n, new_state)

    # ------------------------------------------------------------------ #
    #  Partial trace / reduced density matrix                              #
    # ------------------------------------------------------------------ #

    def reduced_density_matrix(self, keep: list[int]) -> np.ndarray:
        """
        Compute the reduced density matrix by tracing out all qubits NOT in `keep`.
        Returns a complex128 ndarray of shape (2^len(keep), 2^len(keep)).
        """
        keep = sorted(keep)
        trace_out = [i for i in range(self._n) if i not in keep]
        rho = np.outer(self._state, self._state.conj())
        # Reshape to (2, 2, ..., 2) x (2, 2, ..., 2)  [2n dims]
        rho = rho.reshape([2] * (2 * self._n))
        for q in sorted(trace_out, reverse=True):
            rho = np.trace(rho, axis1=q, axis2=q + self._n)
            self._n -= 1  # temporary — reset after
        self._n += len(trace_out)  # restore
        dim_keep = 2 ** len(keep)
        return rho.reshape(dim_keep, dim_keep)

    # ------------------------------------------------------------------ #
    #  Display helpers                                                     #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return f"QubitState(n={self._n}, dim={self.dim})"

    def dirac_notation(self, threshold: float = 1e-6) -> str:
        """
        Return a human-readable Dirac ket representation.

        Example: '(0.707+0.000j)|00⟩ + (0.707+0.000j)|11⟩'
        """
        terms = []
        fmt = f"{{:0{self._n}b}}"
        for i, amp in enumerate(self._state):
            if abs(amp) > threshold:
                ket = fmt.format(i)
                terms.append(f"({amp.real:.3f}{amp.imag:+.3f}j)|{ket}⟩")
        return " + ".join(terms) if terms else "0"

    def bloch_vector(self, qubit_index: int = 0) -> tuple[float, float, float]:
        """
        Compute the Bloch sphere vector (x, y, z) for a single-qubit state
        or the reduced state of qubit `qubit_index`.
        """
        if self._n == 1:
            rho = np.outer(self._state, self._state.conj())
        else:
            rho = self.reduced_density_matrix([qubit_index])

        x = float(2 * rho[0, 1].real)
        y = float(2 * rho[1, 0].imag)
        z = float((rho[0, 0] - rho[1, 1]).real)
        return x, y, z
