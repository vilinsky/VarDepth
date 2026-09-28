import itertools
import tempfile
import unittest
from pathlib import Path
import numpy as np
from quantum_phase import Observation as O,PhaseProblem,exact,canonical,pair_energy,circuit,hybrid,run_sampler,export_qasm

class QuantumPhaseTests(unittest.TestCase):
    def problem(self):
        return PhaseProblem(['a','b','c'],[[O(0,0,.01),O(1,1,.01)],[O(1,0,.01),O(2,0,.01)]]*3)
    def test_oracle_and_label_symmetry(self):
        p=self.problem();r=exact(p);self.assertEqual(r['bits'],[0,1,1])
        for bits in itertools.product([0,1],repeat=3):
            self.assertAlmostEqual(p.full_nll(bits),p.full_nll([1-x for x in bits]))
        scores=[p.full_nll(b) for b in itertools.product([0,1],repeat=3)]
        self.assertAlmostEqual(r['nll'],min(scores));self.assertEqual(r['states'],4)
    def test_edge_sign_and_endianness(self):
        p=self.problem();edges=p.edges()
        self.assertLess(edges[0,1],0);self.assertGreater(edges[1,2],0)
        self.assertLess(pair_energy([0,1,1],edges),pair_energy([0,0,1],edges))
        from qiskit.primitives import StatevectorSampler
        r=run_sampler(p,[.3,.7],StatevectorSampler(seed=31),shots=512)
        self.assertEqual(r['bits'],[0,1,1])
    def test_ising_unitary_matches_cost(self):
        # With beta=0, each basis amplitude has the intended cost phase,
        # modulo one global phase. Detects an RZZ sign or factor-of-two error.
        from qiskit.quantum_info import Statevector
        p=self.problem();e=p.edges();g=.37;state=Statevector.from_instruction(circuit(3,e,[g,0])).data
        scale=max([abs(w) for w in e.values()]+[1.])
        for k in range(8):
            bits=[(k>>j)&1 for j in range(3)]
            self.assertAlmostEqual(abs(state[k]/state[0]-np.exp(-1j*g*pair_energy(bits,e)/scale)),0.,places=12)
    def test_hybrid_and_qasm(self):
        p=self.problem();r=hybrid(p,shots=256,max_evaluations=12)
        self.assertEqual(r['bits'],[0,1,1]);self.assertAlmostEqual(r['hybrid_gap_to_exact'],0.)
        self.assertFalse(r['quantum']['hardware_executed'])
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'phase.qasm';export_qasm(p,r['quantum']['angles'],out)
            self.assertIn('OPENQASM 3.0',out.read_text())
    def test_validation_and_size_limits(self):
        with self.assertRaises(ValueError):PhaseProblem(['a'],[[O(0,0,.01),O(0,1,.01)]]).validate()
        with self.assertRaises(ValueError):exact(PhaseProblem(list(range(19)),[]))

if __name__=='__main__':unittest.main()
