"""Structured turn actor cannot create a wrong-direction/zero equilibrium."""
import math
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from turn_bc import network


class StructuredTurn(unittest.TestCase):
    def test_direction_gain_bounds_and_stop(self):
        torch.set_num_threads(1)
        model=network()
        for heading in (-1.57,-.17,-.061,0.,.061,.17,1.57):
            with self.subTest(heading=heading):
                x=torch.randn(80);x[-2]=math.sin(heading);x[-1]=math.cos(heading)
                w=float(model(x).item())*.6
                self.assertLessEqual(abs(w),.600001)
                if heading:
                    self.assertGreater(w*heading,0.)
                    self.assertGreaterEqual(abs(w)+1e-7,min(.6,.25*abs(heading)))
                else:self.assertEqual(w,0.)

    def test_training_gradient_reaches_actor(self):
        model=network();x=torch.zeros((3,80));x[:,-2]=math.sin(.1);x[:,-1]=math.cos(.1)
        ((model(x)-.25)**2).mean().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()),0.)


if __name__=='__main__':unittest.main()
