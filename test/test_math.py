import pytest
from src.tools.math import angleDiff
import math

def test_anglediffNegative():
   r = angleDiff(0, 3*math.pi/2) 
   assert r == pytest.approx(-math.pi/2)

def test_anglediffPositive():
   r = angleDiff(0, math.pi/2) 
   assert r == pytest.approx(math.pi/2)
