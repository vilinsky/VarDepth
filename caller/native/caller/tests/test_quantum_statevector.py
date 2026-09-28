import tempfile,unittest
from pathlib import Path
import numpy as np
from quantum_phase import circuit,PhaseProblem,Observation,hybrid,export_qasm
from quantum_statevector import native_probabilities,canonical_probabilities,tapered_circuit

class StatevectorTests(unittest.TestCase):
    def test_full_probability_distribution_matches_qiskit(self):
        from qiskit.quantum_info import Statevector
        for n in (2,3,6,9):
            rng=np.random.default_rng(971+n);edges={(i,j):float(rng.normal()) for i in range(n) for j in range(i+1,n)}
            angles=rng.uniform(-1,1,6)
            original=np.abs(Statevector.from_instruction(circuit(n,edges,angles)).data)**2
            reduced=np.abs(Statevector.from_instruction(tapered_circuit(n,edges,angles)).data)**2
            np.testing.assert_allclose(native_probabilities(n,edges,angles,False),original,atol=1e-12,rtol=0)
            np.testing.assert_allclose(native_probabilities(n,edges,angles,True),canonical_probabilities(original),atol=1e-12,rtol=0)
            np.testing.assert_allclose(reduced,canonical_probabilities(original),atol=1e-12,rtol=0)
    def test_fast_hybrid_retains_oracle_gate_and_exports_reduced_circuit(self):
        p=PhaseProblem(['a','b','c'],[[Observation(0,0,.01),Observation(1,1,.01)],[Observation(1,0,.01),Observation(2,0,.01)]]*3)
        result=hybrid(p,simulator='native',shots=256,max_evaluations=12)
        self.assertEqual(result['bits'],[0,1,1]);self.assertAlmostEqual(result['hybrid_gap_to_exact'],0.)
        self.assertEqual(result['quantum']['qubits'],2)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'taper.qasm';export_qasm(p,result['quantum']['angles'],out,taper=True)
            self.assertIn('qubit[2]',out.read_text())

if __name__=='__main__':unittest.main()
