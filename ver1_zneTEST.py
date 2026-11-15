# plot test: fake numbers, no IBM connection

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

# build phase, ghz circuit
circ = QuantumCircuit(3)
circ.h(0)
circ.cx(0, 1)
circ.cx(0, 2)

print(circ.draw())

sv = Statevector(circ)
probs = {str(k): float(v) for k, v in sv.probabilities_dict(decimals=3).items()}
print("Perfect result:", probs)
# if sv.probabilities_dict(decimals=3) errors, use sv.probabilities_dict()

# XXX = "flip all 3 qubits, does the state match itself?"
# +1 = fully entangled, 0 = link lost, -1 = perfect opposite
# H and CNOT gates produce plus ver, circ makes 000 + 111, that why no minus
observable = SparsePauliOp("XXX")

# exact value: what a perfect, noise-free computer gives for XXX
exact = float(sv.expectation_value(observable).real)
print(f"Exact value: {exact:.3f}")

# FAKE numbers standing in for the IBM results
unmitigated = 0.60
mitigated = 0.85
raw_values = np.array([0.60, 0.40, 0.28])
errs = np.array([0.02, 0.02, 0.02])

print(f"Unmitigated: {unmitigated:.3f}")
print(f"Mitigated:   {mitigated:.3f}")
print(f"Exact:       {exact:.3f}")

# plot
noise_factors = [1, 3, 5]
xs = np.linspace(0, 5, 50)

plt.errorbar(noise_factors, raw_values, yerr=errs, fmt="o", capsize=4, label="ZNE noise levels (1x, 3x, 5x)")

if np.all(raw_values > 0):
    b, log_a = np.polyfit(noise_factors, np.log(raw_values), 1)
    plt.plot(xs, np.exp(log_a + b * xs), "--", label="Exponential fit")
else:  # unwanted scenario
    slope, intercept = np.polyfit(noise_factors, raw_values, 1)
    plt.plot(xs, slope * xs + intercept, "--", label="Linear fit (fallback)")

plt.scatter([1], [unmitigated], marker="s", s=100, label="Unmitigated run")
plt.scatter([0], [mitigated], marker="*", s=200, label="Mitigated run (ZNE)")
plt.axhline(exact, color="gray", linestyle=":", label=f"Exact value ({exact:.1f})")

plt.xlabel("Noise level (1 = normal)")
plt.ylabel("XXX expectation value")
plt.title("GHZ with Error Mitigation")
plt.legend()
plt.savefig("plot_test.png")
plt.show()