from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
# translates cir into smth real IBM chip understands
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
# sends circ to IBM machine, returns results
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# build phase, ghz circuit
circ = QuantumCircuit(3)
circ.h(0)
circ.cx(0, 1)
circ.cx(0, 2)

# results produced in perf comp, conf gates b4 q time
print(circ.draw())
statevector = Statevector(circ)
print(statevector)

# stops "spin state" reports results
# must occur after statevector is computed since measuring destroys state
# qiskit no comp on measured circuit
circ.measure_all()
print(circ.draw())

# running on real hardware, chose most busy machine bc following trend
service = QiskitRuntimeService(name="open")
# backends holds list of machines
backends = service.backends(operational=True, simulator=False, min_num_qubits=3)
# backend holds machine with most pending jobs
backend = max(backends, key=lambda b: b.status().pending_jobs)
# backend = service.least_backend = service.least_busy(operational=True, simulator=False, min_num_qubits=3)
print("Running on:", backend.name, "\nJob amt:", backend.status().pending_jobs)

# translates circ
pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
# stands for "Instruction Set Architecture", set of moves chip supports
isa_circuit = pm.run(circ)

# sending circ, receive results
sampler = Sampler(mode=backend)
# 1024 shots bc habit, power of 2
job = sampler.run([isa_circuit], shots=1024)
print("Job ID:", job.job_id())

result = job.result()
# result for first circ (result[0])
print(result[0].data.meas.get_counts())
