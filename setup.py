# QuantumSimulator — setup.py

from setuptools import setup, find_packages

setup(
    name="quantum_simulator",
    version="1.0.0",
    description="A high-fidelity quantum circuit simulator for 1–8 qubits",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
    ],
    extras_require={
        "plot": ["matplotlib>=3.6"],
        "dev":  ["pytest>=7.0", "matplotlib>=3.6"],
    },
)
