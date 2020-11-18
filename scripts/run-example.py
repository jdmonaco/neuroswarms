#!/usr/bin/env python

from neuroswarms.model import NeuroswarmsModel


# Run a 22s simulation with 128 agents and larger spatial constants for swarming
# (sigma) and reward approach (kappa):

model = NeuroswarmsModel()
model.simulate(tag='example', duration=22.0, N_S=128, sigma=2.0, kappa=1.5)


# The results will be saved in `~/neuroswarms/output` by default. This can be
# changed by editing PROJDIR or RUNDIR in `neuroswarms/__init__.py`.
