"""
QuantumSimulator.visualization
================================
Text-based and matplotlib visualisation utilities.

All matplotlib calls are optional (guarded by try/import) so the library
works without a display / GUI.  Callers that want plots should have
matplotlib installed:  pip install matplotlib
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from ..qubit import QubitState
from .circuit_simulator import CircuitSimulator


# =========================================================================== #
#  Text utilities                                                              #
# =========================================================================== #

def state_table(state: QubitState, threshold: float = 1e-6) -> str:
    """
    Return a formatted table of basis states, amplitudes, and probabilities.

    Example output:
        ┌───────┬──────────────────────┬────────────┐
        │ Basis │      Amplitude       │ Probability│
        ├───────┼──────────────────────┼────────────┤
        │  |00⟩ │  0.7071 + 0.0000j   │  50.00 %   │
        │  |11⟩ │  0.7071 + 0.0000j   │  50.00 %   │
        └───────┴──────────────────────┴────────────┘
    """
    n = state.n_qubits
    fmt = f"{{:0{n}b}}"
    rows = []
    for i, amp in enumerate(state.amplitudes):
        prob = abs(amp) ** 2
        if prob > threshold:
            rows.append((fmt.format(i), amp, prob))

    col1 = max(len(r[0]) + 2 for r in rows) + 2  # |xx⟩
    col2 = 24
    col3 = 12

    top    = f"┌{'─'*col1}┬{'─'*col2}┬{'─'*col3}┐"
    header = f"│{'Basis':^{col1}}│{'Amplitude':^{col2}}│{'Probability':^{col3}}│"
    sep    = f"├{'─'*col1}┼{'─'*col2}┼{'─'*col3}┤"
    bot    = f"└{'─'*col1}┴{'─'*col2}┴{'─'*col3}┘"

    lines = [top, header, sep]
    for basis, amp, prob in rows:
        b_cell = f"|{basis}⟩".center(col1)
        a_cell = f"{amp.real:+.4f} {amp.imag:+.4f}j".center(col2)
        p_cell = f"{prob*100:6.2f} %".center(col3)
        lines.append(f"│{b_cell}│{a_cell}│{p_cell}│")
    lines.append(bot)
    return "\n".join(lines)


def histogram_text(counts: dict[str, int], width: int = 40) -> str:
    """ASCII bar chart for measurement counts."""
    total = sum(counts.values())
    if total == 0:
        return "(no results)"
    max_count = max(counts.values())
    lines = [f"Measurement histogram  (shots={total})"]
    lines.append("─" * (width + 16))
    for state, count in sorted(counts.items()):
        bar_len = int(count / max_count * width)
        bar = "█" * bar_len + "░" * (width - bar_len)
        pct = count / total * 100
        lines.append(f" |{state}⟩  {bar}  {count:5d}  ({pct:5.1f}%)")
    lines.append("─" * (width + 16))
    return "\n".join(lines)


# =========================================================================== #
#  Matplotlib plots (optional)                                                #
# =========================================================================== #

def plot_probabilities(state: QubitState, title: str = "State Probabilities") -> None:
    """Bar chart of Born-rule probabilities (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[visualization] matplotlib not installed — using text fallback.")
        print(state_table(state))
        return

    n = state.n_qubits
    fmt = f"{{:0{n}b}}"
    labels = [fmt.format(i) for i in range(state.dim)]
    probs  = state.probabilities

    fig, ax = plt.subplots(figsize=(max(6, state.dim * 0.5), 4))
    colours = plt.cm.plasma(np.linspace(0.15, 0.85, state.dim))
    bars = ax.bar(labels, probs, color=colours, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Basis state", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 1.05)
    for bar, p in zip(bars, probs):
        if p > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, p + 0.02, f"{p:.3f}",
                    ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_histogram(counts: dict[str, int], title: str = "Measurement Counts") -> None:
    """Bar chart of measurement counts (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[visualization] matplotlib not installed — using text fallback.")
        print(histogram_text(counts))
        return

    states = sorted(counts.keys())
    values = [counts[s] for s in states]
    total  = sum(values)

    fig, ax = plt.subplots(figsize=(max(6, len(states) * 0.55), 4))
    colours = plt.cm.viridis(np.linspace(0.15, 0.85, len(states)))
    bars = ax.bar(states, values, color=colours, edgecolor="white")
    ax.set_xlabel("Measured state", fontsize=12)
    ax.set_ylabel("Counts", fontsize=12)
    ax.set_title(f"{title}  (shots={total})", fontsize=14)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                f"{v/total*100:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_bloch_sphere(state: QubitState, qubit: int = 0) -> None:
    """Render Bloch sphere for a single-qubit (or reduced) state."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        x, y, z = state.bloch_vector(qubit)
        print(f"Bloch vector for qubit {qubit}: ({x:.4f}, {y:.4f}, {z:.4f})")
        return

    x, y, z = state.bloch_vector(qubit)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")

    # Draw sphere wireframe
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(u.size), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="lightblue", alpha=0.3, linewidth=0.4)

    # Axes
    for dx, dy, dz, lbl in [(1.2,0,0,"X"),(0,1.2,0,"Y"),(0,0,1.2,"|0⟩"),(0,0,-1.2,"|1⟩")]:
        ax.quiver(0,0,0,dx,dy,dz, color="gray", linewidth=0.8)
        ax.text(dx,dy,dz,lbl,fontsize=11,ha="center")

    # State vector arrow
    ax.quiver(0,0,0,x,y,z, color="#e040fb", linewidth=3, arrow_length_ratio=0.15)
    ax.scatter([x],[y],[z], color="#e040fb", s=80, zorder=5)

    ax.set_title(f"Bloch sphere — qubit {qubit}  ({x:.2f}, {y:.2f}, {z:.2f})", fontsize=12)
    ax.set_box_aspect([1,1,1])
    ax.axis("off")
    plt.tight_layout()
    plt.show()
