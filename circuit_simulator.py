"""
QuantumSimulator.circuit_simulator
===================================
CircuitSimulator — high-level quantum circuit builder.

Builds on top of GateSimulator, adding:
  • Declarative gate scheduling  (circuit is built first, run later)
  • Multi-run statistics
  • ASCII circuit diagram rendering
  • Circuit composition (append, prepend, tensor product)
  • Parametric circuits (variational / ansatz support)
  • Built-in common circuit templates (Bell, GHZ, QFT, Grover, Teleportation)
"""

from __future__ import annotations
import numpy as np
from copy import deepcopy
from typing import Any, Callable, Optional

from .gate_simulator import GateSimulator, GateSimulatorError
from .qubit import QubitState


# =========================================================================== #
#  Instruction dataclass                                                       #
# =========================================================================== #

class Instruction:
    """A single gate instruction (gate name + qubit targets + parameters)."""

    __slots__ = ("gate", "qubits", "params", "label")

    def __init__(
        self,
        gate: str,
        qubits: list[int],
        params: list[float] | None = None,
        label: str | None = None,
    ) -> None:
        self.gate = gate.upper()
        self.qubits = qubits
        self.params = params or []
        self.label = label or gate

    def __repr__(self) -> str:
        p = f", {self.params}" if self.params else ""
        return f"Instruction({self.gate} q{self.qubits}{p})"


# =========================================================================== #
#  CircuitSimulator                                                            #
# =========================================================================== #

class CircuitSimulator:
    """
    High-level quantum circuit simulator.

    Usage
    -----
    >>> circ = CircuitSimulator(2, name="BellCircuit")
    >>> circ.h(0).cnot(0, 1)
    >>> result = circ.run(shots=2048)
    >>> circ.draw()
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        n_qubits: int,
        name: str = "QuantumCircuit",
        initial_state: int | np.ndarray = 0,
    ) -> None:
        if not (1 <= n_qubits <= 8):
            raise GateSimulatorError(
                f"n_qubits must be 1–8, got {n_qubits}."
            )
        self._n = n_qubits
        self.name = name
        self._initial = initial_state
        self._instructions: list[Instruction] = []
        self._last_result: dict[str, int] | None = None

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def n_qubits(self) -> int:
        return self._n

    @property
    def depth(self) -> int:
        """Circuit depth (number of sequential gate layers)."""
        return len(self._instructions)

    @property
    def instructions(self) -> list[Instruction]:
        return list(self._instructions)

    # ------------------------------------------------------------------ #
    #  Gate API — returns self for chaining                                #
    # ------------------------------------------------------------------ #

    def _add(self, gate: str, qubits: list[int], params: list[float] | None = None, label: str | None = None) -> "CircuitSimulator":
        self._instructions.append(Instruction(gate, qubits, params, label))
        return self

    # Single-qubit
    def i(self, q):    return self._add("I",   [q])
    def x(self, q):    return self._add("X",   [q])
    def y(self, q):    return self._add("Y",   [q])
    def z(self, q):    return self._add("Z",   [q])
    def h(self, q):    return self._add("H",   [q])
    def s(self, q):    return self._add("S",   [q])
    def sdg(self, q):  return self._add("SDG", [q])
    def t(self, q):    return self._add("T",   [q])
    def tdg(self, q):  return self._add("TDG", [q])

    def rx(self, q, theta):           return self._add("RX",  [q], [theta])
    def ry(self, q, theta):           return self._add("RY",  [q], [theta])
    def rz(self, q, theta):           return self._add("RZ",  [q], [theta])
    def r1(self, q, lam):             return self._add("R1",  [q], [lam])
    def u3(self, q, theta, phi, lam): return self._add("U3",  [q], [theta, phi, lam])

    # Two-qubit
    def cnot(self, ctrl, tgt):        return self._add("CNOT",  [ctrl, tgt])
    def cx(self, ctrl, tgt):          return self.cnot(ctrl, tgt)
    def cz(self, ctrl, tgt):          return self._add("CZ",    [ctrl, tgt])
    def swap(self, q0, q1):           return self._add("SWAP",  [q0, q1])
    def iswap(self, q0, q1):          return self._add("ISWAP", [q0, q1])
    def ch(self, ctrl, tgt):          return self._add("CH",    [ctrl, tgt])
    def cp(self, ctrl, tgt, lam):     return self._add("CP",    [ctrl, tgt], [lam])

    # Three-qubit
    def toffoli(self, c0, c1, tgt):   return self._add("CCX",   [c0, c1, tgt])
    def ccx(self, c0, c1, tgt):       return self.toffoli(c0, c1, tgt)
    def fredkin(self, ctrl, t0, t1):  return self._add("CSWAP", [ctrl, t0, t1])
    def cswap(self, ctrl, t0, t1):    return self.fredkin(ctrl, t0, t1)

    # Oracle
    def grover_oracle(self, marked: list[int]):
        return self._add("GROVER_ORACLE", [], label="GroverOracle", params=[float(m) for m in marked])

    def phase_oracle(self, f: Callable[[int], int]):
        self._instructions.append(Instruction("PHASE_ORACLE", [], label="PhaseOracle"))
        self._instructions[-1]._f = f  # type: ignore[attr-defined]
        return self

    def diffusion(self):
        return self._add("DIFFUSION", [])

    def had_all(self):
        return self._add("H_ALL", [])

    # ------------------------------------------------------------------ #
    #  Execution                                                           #
    # ------------------------------------------------------------------ #

    def _build_simulator(self) -> GateSimulator:
        """Instantiate a GateSimulator and replay all instructions."""
        sim = GateSimulator(self._n, self._initial)

        for instr in self._instructions:
            g = instr.gate
            q = instr.qubits
            p = instr.params

            if   g == "I":    sim.i(q[0])
            elif g == "X":    sim.x(q[0])
            elif g == "Y":    sim.y(q[0])
            elif g == "Z":    sim.z(q[0])
            elif g == "H":    sim.h(q[0])
            elif g == "S":    sim.s(q[0])
            elif g == "SDG":  sim.sdg(q[0])
            elif g == "T":    sim.t(q[0])
            elif g == "TDG":  sim.tdg(q[0])
            elif g == "RX":   sim.rx(q[0], p[0])
            elif g == "RY":   sim.ry(q[0], p[0])
            elif g == "RZ":   sim.rz(q[0], p[0])
            elif g == "R1":   sim.r1(q[0], p[0])
            elif g == "U3":   sim.u3(q[0], p[0], p[1], p[2])
            elif g == "CNOT": sim.cnot(q[0], q[1])
            elif g == "CZ":   sim.cz(q[0], q[1])
            elif g == "SWAP": sim.swap(q[0], q[1])
            elif g == "ISWAP":sim.iswap(q[0], q[1])
            elif g == "CH":   sim.ch(q[0], q[1])
            elif g == "CP":   sim.cp(q[0], q[1], p[0])
            elif g == "CCX":  sim.toffoli(q[0], q[1], q[2])
            elif g == "CSWAP":sim.fredkin(q[0], q[1], q[2])
            elif g == "GROVER_ORACLE": sim.grover_oracle([int(m) for m in p])
            elif g == "PHASE_ORACLE":  sim.phase_oracle(instr._f)  # type: ignore
            elif g == "DIFFUSION":     sim.diffusion()
            elif g == "H_ALL":         sim.had_all()
            else:
                raise GateSimulatorError(f"Unknown gate: {g}")

        return sim

    def run(
        self,
        shots: int = 1024,
        seed: int | None = None,
    ) -> dict[str, int]:
        """
        Execute the circuit and return measurement counts.

        Parameters
        ----------
        shots : int
            Number of measurement repetitions.
        seed : int or None
            RNG seed.

        Returns
        -------
        dict mapping bitstring → count
        """
        sim = self._build_simulator()
        result = sim.measure(shots=shots, seed=seed)
        self._last_result = result
        return result

    def statevector(self) -> QubitState:
        """Execute the circuit and return the final QubitState."""
        return self._build_simulator().state

    def unitary(self) -> np.ndarray:
        """
        Compute the full unitary matrix of the circuit by running it
        on each computational basis state.
        """
        dim = 2**self._n
        U = np.zeros((dim, dim), dtype=np.complex128)
        for col in range(dim):
            circ_copy = deepcopy(self)
            circ_copy._initial = col
            state = circ_copy.statevector()
            U[:, col] = state.amplitudes
        return U

    # ------------------------------------------------------------------ #
    #  Circuit composition                                                 #
    # ------------------------------------------------------------------ #

    def append(self, other: "CircuitSimulator") -> "CircuitSimulator":
        """Append another circuit (must have same n_qubits)."""
        if other.n_qubits != self._n:
            raise GateSimulatorError(
                f"Cannot append {other.n_qubits}-qubit circuit to {self._n}-qubit circuit."
            )
        self._instructions.extend(deepcopy(other._instructions))
        return self

    def inverse(self) -> "CircuitSimulator":
        """Return the inverse (dagger) of this circuit."""
        inv = CircuitSimulator(self._n, name=f"{self.name}†", initial_state=self._initial)
        _DAGGER = {"X":"X","Y":"Y","Z":"Z","H":"H","S":"SDG","SDG":"S",
                   "T":"TDG","TDG":"T","CNOT":"CNOT","CZ":"CZ","SWAP":"SWAP",
                   "ISWAP":"ISWAP","CCX":"CCX","CSWAP":"CSWAP"}
        for instr in reversed(self._instructions):
            dg = _DAGGER.get(instr.gate)
            if dg:
                inv._add(dg, instr.qubits, instr.params)
            elif instr.gate in ("RX","RY","RZ","R1","CP"):
                inv._add(instr.gate, instr.qubits, [-p for p in instr.params])
            elif instr.gate == "U3":
                theta, phi, lam = instr.params
                inv._add("U3", instr.qubits, [theta, -lam, -phi])
            else:
                raise GateSimulatorError(f"Cannot automatically invert gate: {instr.gate}")
        return inv

    def repeat(self, times: int) -> "CircuitSimulator":
        """Return a new circuit with this circuit repeated `times` times."""
        repeated = CircuitSimulator(self._n, name=f"{self.name}×{times}", initial_state=self._initial)
        for _ in range(times):
            repeated._instructions.extend(deepcopy(self._instructions))
        return repeated

    # ------------------------------------------------------------------ #
    #  ASCII circuit diagram                                               #
    # ------------------------------------------------------------------ #

    def draw(self) -> str:
        """
        Render an ASCII-art circuit diagram.

        Returns the string and also prints it.
        """
        n = self._n
        # Each instruction becomes a column
        cols = []
        for instr in self._instructions:
            col = ["─"] * n  # default wire
            g = instr.gate
            q = instr.qubits

            if g in ("I","X","Y","Z","H","S","SDG","T","TDG"):
                col[q[0]] = f"[{g}]"
            elif g in ("RX","RY","RZ","R1"):
                col[q[0]] = f"[{g}]"
            elif g == "U3":
                col[q[0]] = "[U3]"
            elif g == "CNOT":
                col[q[0]] = "[●]"
                col[q[1]] = "[⊕]"
            elif g == "CZ":
                col[q[0]] = "[●]"
                col[q[1]] = "[Z]"
            elif g == "SWAP":
                col[q[0]] = "[✕]"
                col[q[1]] = "[✕]"
            elif g == "ISWAP":
                col[q[0]] = "[i✕]"
                col[q[1]] = "[i✕]"
            elif g == "CH":
                col[q[0]] = "[●]"
                col[q[1]] = "[H]"
            elif g == "CP":
                col[q[0]] = "[●]"
                col[q[1]] = "[P]"
            elif g == "CCX":
                col[q[0]] = "[●]"
                col[q[1]] = "[●]"
                col[q[2]] = "[⊕]"
            elif g == "CSWAP":
                col[q[0]] = "[●]"
                col[q[1]] = "[✕]"
                col[q[2]] = "[✕]"
            elif g == "GROVER_ORACLE":
                for i in range(n): col[i] = "[Oω]"
            elif g == "PHASE_ORACLE":
                for i in range(n): col[i] = "[Of]"
            elif g == "DIFFUSION":
                for i in range(n): col[i] = "[D]"
            elif g == "H_ALL":
                for i in range(n): col[i] = "[H]"

            cols.append(col)

        # Determine cell width
        max_w = max(
            max((len(cell) for cell in col), default=1)
            for col in cols
        ) if cols else 3

        lines = []
        for q in range(n):
            label = f"q{q} "
            row = label
            for col in cols:
                cell = col[q]
                row += cell.center(max_w, "─")
            row += "─ ⟩"
            lines.append(row)

        diagram = "\n".join(lines)
        header = f"\n  Circuit: {self.name}  ({n} qubits, depth={self.depth})\n"
        full = header + diagram + "\n"
        print(full)
        return full

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def clear(self) -> "CircuitSimulator":
        """Remove all instructions."""
        self._instructions.clear()
        self._last_result = None
        return self

    def clone(self) -> "CircuitSimulator":
        """Return a deep copy."""
        return deepcopy(self)

    def __repr__(self) -> str:
        return f"CircuitSimulator('{self.name}', n={self._n}, depth={self.depth})"

    # ------------------------------------------------------------------ #
    #  Built-in circuit templates                                          #
    # ------------------------------------------------------------------ #

    @classmethod
    def bell_state(cls, pair: int = 0) -> "CircuitSimulator":
        """
        Generate one of the four Bell states on 2 qubits.

        pair : 0 → Φ+, 1 → Φ−, 2 → Ψ+, 3 → Ψ−
        """
        circ = cls(2, name=f"Bell(Φ{pair})")
        if pair in (1, 3): circ.x(0)
        if pair in (2, 3): circ.x(1)
        circ.h(0).cnot(0, 1)
        return circ

    @classmethod
    def ghz_state(cls, n: int) -> "CircuitSimulator":
        """GHZ state on n qubits."""
        circ = cls(n, name=f"GHZ({n})")
        circ.h(0)
        for i in range(n - 1):
            circ.cnot(i, i + 1)
        return circ

    @classmethod
    def qft(cls, n: int) -> "CircuitSimulator":
        """
        Quantum Fourier Transform on n qubits.
        """
        circ = cls(n, name=f"QFT({n})")
        for j in range(n):
            circ.h(j)
            for k in range(j + 1, n):
                lam = np.pi / (2 ** (k - j))
                circ.cp(k, j, lam)
        # Swap to correct output order
        for i in range(n // 2):
            circ.swap(i, n - 1 - i)
        return circ

    @classmethod
    def grover(cls, n: int, marked: list[int], iterations: int | None = None) -> "CircuitSimulator":
        """
        Grover's search algorithm.

        Parameters
        ----------
        n : int
            Number of qubits.
        marked : list[int]
            Indices of marked states.
        iterations : int or None
            Number of Grover iterations (default: optimal ≈ π/4 √N).
        """
        N = 2**n
        if iterations is None:
            iterations = max(1, int(round(np.pi / 4 * np.sqrt(N / len(marked)))))
        circ = cls(n, name=f"Grover(n={n}, iters={iterations})")
        circ.had_all()
        for _ in range(iterations):
            circ.grover_oracle(marked)
            circ.diffusion()
        return circ

    @classmethod
    def teleportation(cls) -> "CircuitSimulator":
        """
        3-qubit quantum teleportation circuit.
        q0 = message qubit (prepared in some state before running).
        q1,q2 = Bell pair.
        """
        circ = cls(3, name="Teleportation")
        # Prepare Bell pair on q1,q2
        circ.h(1).cnot(1, 2)
        # Bell measurement on q0,q1
        circ.cnot(0, 1).h(0)
        # Classically-controlled corrections
        circ.cnot(1, 2).cz(0, 2)
        return circ

    @classmethod
    def w_state(cls, n: int) -> "CircuitSimulator":
        """
        W-state on n qubits.
        |W_n⟩ = (|100...0⟩ + |010...0⟩ + ... + |000...1⟩) / √n
        Built recursively using controlled-Ry rotations.
        """
        circ = cls(n, name=f"W({n})")
        circ.x(0)
        for k in range(1, n):
            theta = 2 * np.arccos(np.sqrt(1 / (n - k + 1)))
            # Controlled-Ry: approximate using decomposition
            circ.ry(k, theta)
            circ.cnot(k - 1, k)
            circ.ry(k, -theta)
        return circ
