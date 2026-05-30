from Cosmika.qubit import QubitState,QuantumStateError
from Cosmika.gate_simulator import GateSimulator,CircuitSimulator,GateSimulatorError

from Cosmika.gate_simulator.visualization import state_table,histogram_text
from Cosmika import gates as G
import pytest

import numpy as np
class TestQubitState:

    def test_defaults_to_zero_state(self):
        q = QubitState(3)
        assert q.amplitudes[0] == pytest.approx(1.0)
        assert all(q.amplitudes[1:] == 0)

    def test_basis_state_init(self):
        q = QubitState(2, 3)   # |11⟩
        assert q.amplitudes[3] == pytest.approx(1.0)

    def test_invalid_n_qubits(self):
        with pytest.raises(QuantumStateError):
            QubitState(0)
        with pytest.raises(QuantumStateError):
            QubitState(9)

    def test_invalid_basis_index(self):
        with pytest.raises(QuantumStateError):
            QubitState(2, 4)  # max index is 3

    def test_unnormalised_raises(self):
        vec = np.array([1.0, 1.0], dtype=complex)
        with pytest.raises(QuantumStateError):
            QubitState(1, vec)

    def test_custom_vector(self):
        vec = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        q = QubitState(1, vec)
        assert np.isclose(np.linalg.norm(q.amplitudes), 1.0)

    def test_probabilities_sum_to_one(self):
        vec = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        q = QubitState(2, vec)
        assert np.isclose(q.probabilities.sum(), 1.0)

    def test_measure_returns_valid_counts(self):
        q = QubitState(1)
        counts = q.measure(shots=100, seed=0)
        assert "0" in counts
        assert sum(counts.values()) == 100

    def test_dirac_notation_zero_state(self):
        q = QubitState(2)
        assert "|00⟩" in q.dirac_notation()

    def test_bloch_vector_zero_state(self):
        q = QubitState(1)
        x, y, z = q.bloch_vector()
        assert np.isclose(z, 1.0)
        assert np.isclose(x, 0.0)
        assert np.isclose(y, 0.0)

    def test_bloch_vector_one_state(self):
        q = QubitState(1, 1)
        x, y, z = q.bloch_vector()
        assert np.isclose(z, -1.0)


# =========================================================================== #
#  Gate matrices                                                               #
# =========================================================================== #

class TestGates:

    def _assert_unitary(self, m):
        assert np.allclose(m @ m.conj().T, np.eye(m.shape[0]), atol=1e-9)

    def test_all_single_qubit_unitary(self):
        for name, mat in G.SINGLE_QUBIT_GATES.items():
            self._assert_unitary(mat)

    def test_rx_ry_rz_unitary(self):
        for theta in [0, np.pi/4, np.pi/2, np.pi, 2*np.pi]:
            self._assert_unitary(G.rx(theta))
            self._assert_unitary(G.ry(theta))
            self._assert_unitary(G.rz(theta))

    def test_cnot_unitary(self):      self._assert_unitary(G.cnot())
    def test_cz_unitary(self):        self._assert_unitary(G.cz())
    def test_swap_unitary(self):      self._assert_unitary(G.swap())
    def test_iswap_unitary(self):     self._assert_unitary(G.iswap())
    def test_toffoli_unitary(self):   self._assert_unitary(G.toffoli())
    def test_fredkin_unitary(self):   self._assert_unitary(G.fredkin())

    def test_hadamard_is_its_own_inverse(self):
        h = G.hadamard()
        assert np.allclose(h @ h, np.eye(2), atol=1e-9)

    def test_pauli_x_flips_bit(self):
        x = G.pauli_x()
        v = np.array([1.0, 0.0], dtype=complex)
        result = x @ v
        assert np.allclose(result, [0, 1])

    def test_t_s_relationship(self):
        # S = T^2
        t = G.t_gate()
        s = G.phase_s()
        assert np.allclose(t @ t, s, atol=1e-9)

    def test_grover_oracle_marks_state(self):
        oracle = G.grover_oracle([3], 2)
        assert oracle[3, 3] == pytest.approx(-1.0)
        assert oracle[0, 0] == pytest.approx(1.0)

    def test_phase_oracle_function(self):
        f = lambda x: 1 if x == 2 else 0
        oracle = G.phase_oracle(f, 2)
        assert oracle[2, 2] == pytest.approx(-1.0)
        assert oracle[0, 0] == pytest.approx(1.0)

    def test_u3_general_unitary(self):
        self._assert_unitary(G.u3(np.pi/3, np.pi/4, np.pi/5))


# =========================================================================== #
#  GateSimulator                                                               #
# =========================================================================== #

class TestGateSimulator:

    def test_initial_state(self):
        sim = GateSimulator(3)
        assert np.isclose(sim.state.amplitudes[0], 1.0)

    def test_pauli_x_flip(self):
        sim = GateSimulator(1)
        sim.x(0)
        assert np.isclose(sim.state.amplitudes[1], 1.0)

    def test_hadamard_superposition(self):
        sim = GateSimulator(1)
        sim.h(0)
        probs = sim.state.probabilities
        assert np.isclose(probs[0], 0.5, atol=1e-6)
        assert np.isclose(probs[1], 0.5, atol=1e-6)

    def test_bell_state(self):
        sim = GateSimulator(2)
        sim.h(0).cnot(0, 1)
        amps = sim.state.amplitudes
        assert np.isclose(abs(amps[0]), 1/np.sqrt(2))
        assert np.isclose(abs(amps[3]), 1/np.sqrt(2))
        assert np.isclose(abs(amps[1]), 0)
        assert np.isclose(abs(amps[2]), 0)

    def test_ghz_state(self):
        sim = GateSimulator(3)
        sim.h(0).cnot(0, 1).cnot(0, 2)
        amps = sim.state.amplitudes
        assert np.isclose(abs(amps[0]), 1/np.sqrt(2))
        assert np.isclose(abs(amps[7]), 1/np.sqrt(2))

    def test_toffoli_gate(self):
        # |110⟩ → |111⟩
        sim = GateSimulator(3, 6)  # |110⟩ = index 6
        sim.toffoli(0, 1, 2)
        assert np.isclose(sim.state.amplitudes[7], 1.0)  # |111⟩ = index 7

    def test_fredkin_gate(self):
        # Control=1, swap |110⟩ qubits 1,2 → |101⟩
        sim = GateSimulator(3, 6)  # |110⟩
        sim.fredkin(0, 1, 2)
        assert np.isclose(sim.state.amplitudes[5], 1.0)  # |101⟩ = 5

    def test_swap_gate(self):
        sim = GateSimulator(2, 1)  # |01⟩
        sim.swap(0, 1)
        assert np.isclose(sim.state.amplitudes[2], 1.0)  # |10⟩ = 2

    def test_rx_rotation(self):
        sim = GateSimulator(1)
        sim.rx(0, np.pi)      # Rx(π) ≈ -iX  → |1⟩ (up to global phase)
        assert np.isclose(sim.state.probabilities[1], 1.0, atol=1e-6)

    def test_history_tracking(self):
        sim = GateSimulator(2)
        sim.h(0).cnot(0, 1).z(1)
        assert len(sim.history) == 3

    def test_snapshot_restore(self):
        sim = GateSimulator(1)
        sim.snapshot()
        sim.x(0)
        assert np.isclose(sim.state.probabilities[1], 1.0)
        sim.restore()
        assert np.isclose(sim.state.probabilities[0], 1.0)

    def test_invalid_qubit_index(self):
        sim = GateSimulator(2)
        with pytest.raises(GateSimulatorError):
            sim.x(5)

    def test_custom_gate_non_unitary_raises(self):
        sim = GateSimulator(1)
        with pytest.raises(GateSimulatorError):
            sim.custom_gate(0, np.array([[2, 0],[0, 0.5]]))

    def test_grover_oracle_in_sim(self):
        sim = GateSimulator(2)
        sim.had_all().grover_oracle([3])  # Mark |11⟩
        probs = sim.state.probabilities
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)

    def test_phase_oracle_in_sim(self):
        f = lambda x: 1 if x == 0 else 0
        sim = GateSimulator(2)
        sim.had_all().phase_oracle(f)
        probs = sim.state.probabilities
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)

    def test_t_tdg_cancel(self):
        sim = GateSimulator(1)
        sim.x(0).t(0).tdg(0)
        assert np.isclose(sim.state.amplitudes[1], 1.0, atol=1e-6)

    def test_s_sdg_cancel(self):
        sim = GateSimulator(1)
        sim.x(0).s(0).sdg(0)
        assert np.isclose(sim.state.amplitudes[1], 1.0, atol=1e-6)

    def test_measure_counts(self):
        sim = GateSimulator(1)
        sim.h(0)
        counts = sim.measure(shots=10000, seed=42)
        assert abs(counts.get("0", 0) - counts.get("1", 0)) < 500


# =========================================================================== #
#  CircuitSimulator                                                            #
# =========================================================================== #

class TestCircuitSimulator:

    def test_run_returns_dict(self):
        circ = CircuitSimulator(2)
        circ.h(0).cnot(0, 1)
        counts = circ.run(shots=100, seed=0)
        assert isinstance(counts, dict)
        assert sum(counts.values()) == 100

    def test_bell_state_counts(self):
        circ = CircuitSimulator.bell_state()
        counts = circ.run(shots=10000, seed=0)
        assert set(counts.keys()) == {"00", "11"}
        assert abs(counts["00"] - counts["11"]) < 500

    def test_ghz_3_state(self):
        circ = CircuitSimulator.ghz_state(3)
        counts = circ.run(shots=1000, seed=0)
        assert set(counts.keys()).issubset({"000", "111"})

    def test_depth(self):
        circ = CircuitSimulator(2)
        circ.h(0).cnot(0, 1).z(1)
        assert circ.depth == 3

    def test_statevector(self):
        circ = CircuitSimulator(1)
        circ.x(0)
        sv = circ.statevector()
        assert np.isclose(sv.amplitudes[1], 1.0)

    def test_unitary_identity(self):
        circ = CircuitSimulator(1)   # No gates = identity
        U = circ.unitary()
        assert np.allclose(U, np.eye(2), atol=1e-6)

    def test_append(self):
        c1 = CircuitSimulator(1)
        c1.h(0)
        c2 = CircuitSimulator(1)
        c2.x(0)
        c1.append(c2)
        assert c1.depth == 2

    def test_inverse_bell(self):
        circ = CircuitSimulator.bell_state()
        inv  = circ.inverse()
        # Bell then inverse Bell should return to |00⟩
        combined = circ.clone().append(inv)
        sv = combined.statevector()
        assert np.isclose(sv.probabilities[0], 1.0, atol=1e-6)

    def test_grover_4qubit(self):
        marked = [7]
        circ = CircuitSimulator.grover(4, marked, iterations=4)
        counts = circ.run(shots=1000, seed=0)
        # State 7 = |0111⟩ should dominate
        assert counts.get("0111", 0) > 400  # marked state dominates

    def test_qft_2qubit_unitary(self):
        circ = CircuitSimulator.qft(2)
        U = circ.unitary()
        # QFT should be unitary
        assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-6)

    def test_draw_returns_string(self):
        circ = CircuitSimulator.bell_state()
        result = circ.draw()
        assert isinstance(result, str)
        assert "H" in result


# =========================================================================== #
#  Visualisation utilities                                                     #
# =========================================================================== #

class TestVisualization:

    def test_state_table_returns_string(self):
        q = QubitState(2)
        table = state_table(q)
        assert isinstance(table, str)
        assert "|00⟩" in table

    def test_histogram_text_returns_string(self):
        counts = {"00": 512, "11": 512}
        hist = histogram_text(counts)
        assert "00" in hist
        assert "11" in hist


# =========================================================================== #
#  Integration — multi-step algorithms                                        #
# =========================================================================== #

class TestIntegration:

    def test_deutsch_jozsa_constant(self):
        """Constant function f(x)=0: all-zero measurement."""
        n = 3
        sim = GateSimulator(n)
        sim.had_all()
        # Constant oracle = identity, do nothing
        sim.had_all()
        counts = sim.measure(shots=256, seed=0)
        # Should always measure |000⟩
        assert counts.get("000", 0) == 256

    def test_deutsch_jozsa_balanced(self):
        """Balanced oracle (f flips bit on half states): non-zero measurement."""
        n = 2
        sim = GateSimulator(n)
        sim.had_all()
        # Balanced oracle: phase-flip states 1 and 3
        sim.phase_oracle(lambda x: x % 2)
        sim.had_all()
        counts = sim.measure(shots=256, seed=0)
        # Should NOT measure |00⟩
        assert counts.get("00", 0) == 0

    def test_8_qubit_state_vector_correct_dim(self):
        sim = GateSimulator(8)
        sim.had_all()
        assert sim.state.dim == 256
        assert np.isclose(sim.state.probabilities.sum(), 1.0)

    def test_parametric_circuit(self):
        """Variational circuit with runtime angles."""
        theta = np.pi / 3
        circ = CircuitSimulator(1)
        circ.ry(0, theta)
        sv = circ.statevector()
        expected_0 = np.cos(theta/2)**2
        assert np.isclose(sv.probabilities[0], expected_0, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])