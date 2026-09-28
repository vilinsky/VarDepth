import unittest
from types import SimpleNamespace
from ownership import components
def r(c,s,e):
    x=SimpleNamespace(chrom=c,start=s,stop=e);x.copy=lambda:x;return x
class StreamComponents(unittest.TestCase):
    def test_touching_and_nested(self):
        self.assertEqual([len(x) for x in components([r('a',1,100),r('a',3,4),r('a',100,110),r('a',111,120)])],[3,1])
    def test_chromosome_boundary(self):
        self.assertEqual([len(x) for x in components([r('a',1,100),r('b',1,10)])],[1,1])
    def test_unsorted_fails(self):
        with self.assertRaises(ValueError):list(components([r('a',10,20),r('a',1,2)]))
    def test_repeated_chromosome_fails(self):
        with self.assertRaises(ValueError):list(components([r('a',1,2),r('b',1,2),r('a',3,4)]))
    def test_empty(self):self.assertEqual(list(components([])),[])
if __name__=='__main__':unittest.main()
