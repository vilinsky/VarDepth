"""Behavior tests for the precision/coverage tradeoff, not empirical accuracy."""
import itertools,unittest
import numpy as np
from hybrid_phase import Edge,phase_forest,molecular_edges,phase_edges
from phase_skip39 import phase_edges_bridged

class CoverageTests(unittest.TestCase):
    def test_bridge_extends_block_but_respects_budget(self):
        edges=[Edge(0,1,0,.001,'reads',3),Edge(2,3,1,.001,'reads',3),Edge(1,2,1,.015,'panel')]
        small,_=phase_forest(4,edges,.01);large,_=phase_forest(4,edges,.02)
        self.assertNotEqual(small[0][0],small[3][0]);self.assertEqual(large[0][0],large[3][0])
        self.assertEqual(large[0][1]^large[3][1],0)
        self.assertAlmostEqual(large[0][2],.017)
    def test_two_molecules_allowed_one_withheld(self):
        m=[{0:(0,.005),1:(1,.005)}]
        self.assertEqual(molecular_edges(m,2),[])
        self.assertEqual(len(molecular_edges(m*2,2)),1)
        self.assertEqual(molecular_edges(m*2,3),[])
    def test_conflicting_cycle_still_quarantined(self):
        edges=[Edge(0,1,0,.0001,'reads'),Edge(1,2,0,.0001,'reads'),Edge(0,2,1,.015,'panel')]
        out,stats=phase_forest(3,edges,.02)
        self.assertEqual(out,{});self.assertEqual(stats['withheld_conflict_sites'],3)
    def test_distant_branches_do_not_exceed_pair_budget(self):
        edges=[Edge(0,i,0,.015,'panel') for i in range(1,5)]
        out,_=phase_forest(5,edges,.02)
        self.assertEqual(len(out),2)
    def test_edge_order_does_not_change_output(self):
        edges=[Edge(0,1,0,.001,'reads'),Edge(2,3,1,.001,'reads'),Edge(1,2,1,.015,'panel')]
        self.assertEqual(phase_forest(4,edges,.02),phase_forest(4,list(reversed(edges)),.02))

class InformativeAnchorTests(unittest.TestCase):
    def test_uninformative_heterozygote_not_forced(self):
        rows=phase_edges_bridged([[0,1],[0,0],[0,1]],[1,1,1],[0,.001,.001])
        result={(a,b):p for a,b,p in rows}
        self.assertAlmostEqual(result[0,1],.5);self.assertAlmostEqual(result[1,2],.5)
        self.assertLess(result[0,2],.01)
    def test_all_informative_equals_original(self):
        h=[[0,1,0],[1,0,0],[0,1,1],[1,1,0]];g=[1,1,0,1];r=[0,.01,.02,.005]
        self.assertEqual(phase_edges_bridged(h,g,r),phase_edges(h,g,r))
    def test_bridge_matches_full_latent_state_enumeration(self):
        h=np.array([[0,1],[0,0],[0,1]]);refresh=[0,.03,.07];mutation=.001
        x=mutation+(1-2*mutation)*h;total=discordant=0.
        states=list(itertools.product(range(2),repeat=2))
        for path in itertools.product(states,repeat=3):
            prior=.25
            for t in [1,2]:
                for hap in [0,1]:prior*=refresh[t]/2+(1-refresh[t])*(path[t-1][hap]==path[t][hap])
            for bits in itertools.product([0,1],repeat=3):
                weight=prior
                for t,(a,b) in enumerate(path):
                    weight*=(x[t,a] if bits[t] else 1-x[t,a])*(1-x[t,b] if bits[t] else x[t,b])
                total+=weight;discordant+=weight*(bits[0]!=bits[2])
        result={(a,b):p for a,b,p in phase_edges_bridged(h,[1,1,1],refresh,mutation)}
        self.assertAlmostEqual(result[0,2],discordant/total,places=12)
if __name__=='__main__':unittest.main()
