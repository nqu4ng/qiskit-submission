# local ZNE with Mitiq on a simulated noisy chip: no IBM, no queue

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from mitiq import zne
from mitiq.zne.inference import ExpFactory
from mitiq.zne.scaling import fold_global

SHOTS = 20000

# ghz circuit
circ = QuantumCircuit(3)
circ.h(0)
circ.cx(0, 1)
circ.cx(0, 2)
print(circ.draw())

# exact value of XXX on a perfect computer
observable = SparsePauliOp("XXX")
exact = float(Statevector(circ).expectation_value(observable).real)
print(f"Exact value: {exact:.3f}")

# fake noisy chip: every gate has a small chance of scrambling the qubits
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.004, 1), ["h"])
noise_model.add_all_qubit_quantum_error(depolarizing_error(0.06, 2), ["cx"])
sim = AerSimulator(noise_model=noise_model)

def run_xxx(circuit):
    """Run a circuit on the noisy simulator and return its XXX score."""
    c = circuit.copy()
    c.h(range(3))        # rotate so reading Z gives the X answer
    c.measure_all()
    c = transpile(c, sim, optimization_level=0)   # level 0: don't cancel the folded gates
    counts = sim.run(c, shots=SHOTS).result().get_counts()
    total = sum(counts.values())
    # even number of 1s counts as +1, odd as -1
    return sum((-1) ** (bits.count("1") % 2) * n for bits, n in counts.items()) / total

# unmitigated: one normal noisy run
unmitigated = run_xxx(circ)

# zne: fold the circuit to 1x, 3x, 5x noise, then extrapolate to zero
noise_factors = [1, 3, 5]
fac = ExpFactory(scale_factors=noise_factors, asymptote=0.0)
mitigated = zne.execute_with_zne(circ, run_xxx, factory=fac, scale_noise=fold_global)

raw_values = np.array(fac.get_expectation_values())
errs = np.sqrt(np.clip(1 - raw_values**2, 0, None) / SHOTS)   # shot-noise error bars

print(f"Unmitigated: {unmitigated:.3f}")
print(f"Mitigated:   {mitigated:.3f}")
print(f"Exact:       {exact:.3f}")

# did it help?
margin = 0.1   # pick your own tolerance
err_plain = abs(exact - unmitigated)
err_zne = abs(exact - mitigated)
print(f"Error without ZNE: {err_plain:.3f}")
print(f"Error with ZNE:    {err_zne:.3f}")
print("Improved:", err_zne < err_plain)
print("Within margin:", err_zne <= margin)

# plot
xs = np.linspace(0, 5, 50)
plt.errorbar(noise_factors, raw_values, yerr=errs, fmt="o", capsize=4, label="ZNE noise levels (1x, 3x, 5x)")

if np.all(raw_values > 0):
    b, log_a = np.polyfit(noise_factors, np.log(raw_values), 1)
    plt.plot(xs, np.exp(log_a + b * xs), "--", label="Exponential fit")
else:
    slope, intercept = np.polyfit(noise_factors, raw_values, 1)
    plt.plot(xs, slope * xs + intercept, "--", label="Linear fit (fallback)")

plt.scatter([1], [unmitigated], marker="s", s=100, label="Unmitigated run")
plt.scatter([0], [mitigated], marker="*", s=200, label="Mitigated run (ZNE)")
plt.axhline(exact, color="gray", linestyle=":", label=f"Exact value ({exact:.1f})")

plt.xlabel("Noise level (1 = normal)")
plt.ylabel("XXX expectation value")
plt.title("GHZ with Error Mitigation (simulated noise)")
plt.legend()
plt.savefig("zne_local_plot.png")
plt.show()