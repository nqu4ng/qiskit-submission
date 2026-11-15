# GHZ error mitigation study: circuit size x technique x repeated runs (local, simulated noise)

import warnings
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from mitiq import zne
from mitiq.zne.inference import ExpFactory
from mitiq.zne.scaling import fold_global

warnings.filterwarnings("ignore", message="The input circuit is very short")

# ---- settings you can change ----
SIZES = [3, 4, 5, 6, 8]          # number of qubits to test
REPEATS = 10                     # repeats per size (use 3 for a quick first test)
SHOTS = 4000
CX_ERROR = 0.06                  # two-qubit gate noise
H_ERROR = 0.004                  # one-qubit gate noise
READOUT_ERROR = 0.03             # chance a measured bit gets flipped
NOISE_FACTORS = [1, 3, 5]

rng = np.random.default_rng(42)  # fixed seed so reruns give the same numbers

# ---- fake noisy chip ----
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(depolarizing_error(H_ERROR, 1), ["h"])
noise_model.add_all_qubit_quantum_error(depolarizing_error(CX_ERROR, 2), ["cx"])
noise_model.add_all_qubit_readout_error(
    ReadoutError([[1 - READOUT_ERROR, READOUT_ERROR],
                  [READOUT_ERROR, 1 - READOUT_ERROR]])
)
sim = AerSimulator(noise_model=noise_model)


def make_ghz(n):
    c = QuantumCircuit(n)
    c.h(0)
    for i in range(n - 1):
        c.cx(i, i + 1)
    return c


def run_parity(circuit):
    """Run on the noisy simulator and return the X...X score (+1 = perfect)."""
    c = circuit.copy()
    c.h(range(c.num_qubits))      # rotate so reading Z gives the X answer
    c.measure_all()
    c = transpile(c, sim, optimization_level=0)   # level 0: keep folded gates
    seed = int(rng.integers(1_000_000_000))
    counts = sim.run(c, shots=SHOTS, seed_simulator=seed).result().get_counts()
    total = sum(counts.values())
    # even number of 1s counts as +1, odd as -1
    return sum((-1) ** (bits.count("1") % 2) * k for bits, k in counts.items()) / total


def readout_scale(n):
    """Measure how much readout errors shrink the score, using an all-zeros circuit."""
    c = QuantumCircuit(n)
    c.measure_all()
    c = transpile(c, sim, optimization_level=0)
    seed = int(rng.integers(1_000_000_000))
    counts = sim.run(c, shots=20000, seed_simulator=seed).result().get_counts()
    ones = sum(bits.count("1") * k for bits, k in counts.items())
    p = ones / (n * sum(counts.values()))
    return (1 - 2 * p) ** n


def zne_value(circuit, executor):
    """ZNE with an exponential fit. Returns nan if the fit fails."""
    try:
        fac = ExpFactory(scale_factors=NOISE_FACTORS, asymptote=0.0)
        return zne.execute_with_zne(circuit, executor, factory=fac, scale_noise=fold_global)
    except Exception:
        return np.nan


methods = ["Unmitigated", "Readout only", "ZNE only", "ZNE + readout"]
errors = {m: [] for m in methods}   # errors[method][size index] = list over repeats

for n in SIZES:
    circ = make_ghz(n)
    exact = float(Statevector(circ).expectation_value(SparsePauliOp("X" * n)).real)
    scale = readout_scale(n)
    corrected = lambda c, s=scale: run_parity(c) / s

    per = {m: [] for m in methods}
    for _ in range(REPEATS):
        r = run_parity(circ)
        per["Unmitigated"].append(abs(exact - r))
        per["Readout only"].append(abs(exact - r / scale))
        per["ZNE only"].append(abs(exact - zne_value(circ, run_parity)))
        per["ZNE + readout"].append(abs(exact - zne_value(circ, corrected)))

    for m in methods:
        errors[m].append(per[m])
    fails = int(np.isnan(per["ZNE only"]).sum() + np.isnan(per["ZNE + readout"]).sum())
    print(f"n={n}: done (exact={exact:.2f}, failed ZNE fits: {fails})")

# ---- results table: mean error ± spread over repeats ----
print()
print(f"{'qubits':>6} " + " ".join(f"{m:>16}" for m in methods))
for i, n in enumerate(SIZES):
    cells = [f"{np.nanmean(errors[m][i]):.3f}±{np.nanstd(errors[m][i]):.3f}" for m in methods]
    print(f"{n:>6} " + " ".join(f"{c:>16}" for c in cells))

# ---- plot ----
for m in methods:
    means = [np.nanmean(errors[m][i]) for i in range(len(SIZES))]
    stds = [np.nanstd(errors[m][i]) for i in range(len(SIZES))]
    plt.errorbar(SIZES, means, yerr=stds, marker="o", capsize=4, label=m)

plt.xlabel("Number of qubits in GHZ state")
plt.ylabel("Error vs exact value (lower is better)")
plt.title(f"GHZ error by technique ({REPEATS} runs per point, simulated noise)")
plt.legend()
plt.savefig("ghz_scaling.png")
plt.show()