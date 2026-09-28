import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'base'))
from row_patterns import unique_rows
class RowPatterns(unittest.TestCase):
    def test_all_return_modes(self):
        rng=np.random.default_rng(551)
        for columns in [1,2,7,11,32,63,64,65]:
            for rows in [0,1,31,1000]:
                x=rng.integers(0,2,(rows,columns)).astype(float)
                for inverse,counts in [(False,False),(True,False),(False,True),(True,True)]:
                    old=np.unique(x,axis=0,return_counts=counts,return_inverse=inverse)
                    new=unique_rows(x,return_counts=counts,return_inverse=inverse)
                    if inverse or counts:
                        for a,b in zip(old,new):np.testing.assert_array_equal(a,b)
                    else:np.testing.assert_array_equal(old,new)
                if rows:x[0]=.25
                np.testing.assert_array_equal(np.unique(x,axis=0),unique_rows(x))
if __name__=='__main__':unittest.main()
