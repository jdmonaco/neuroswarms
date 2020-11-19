# NeuroSwarms Example Model: Source Code

This package contains the source code for a neural swarming controller model
that supports simulations of both multi-agent swarming and single-entity
navigation (based on the activity of an internal 'mental' swarm of virtual
particles). Findings based on research using this code were published in the
following paper that appeared in the *Biological Cybernetics* Special Issue
on neuroscience-inspired robotics for navigation in complex environments.

[Monaco, J.D., Hwang, G.M., Schultz, K.M., and Zhang, K. (2020). Cognitive
swarming in complex environments with attractor dynamics and oscillatory
computing. Biol Cybern 114, 269–284.](https://doi.org/10.1007/s00422-020-00823-z)

**Abstract**

> Neurobiological theories of spatial cognition developed with respect to
> recording data from relatively small and/or simplistic environments compared
> to animals’ natural habitats. It has been unclear how to extend theoretical
> models to large or complex spaces. Complementarily, in autonomous systems
> technology, applications have been growing for distributed control methods
> that scale to large numbers of low-footprint mobile platforms. Animals and
> many-robot groups must solve common problems of navigating complex and
> uncertain environments. Here, we introduce the NeuroSwarms control framework
> to investigate whether adaptive, autonomous swarm control of minimal
> artificial agents can be achieved by direct analogy to neural circuits of
> rodent spatial cognition. NeuroSwarms analogizes agents to neurons and
> swarming groups to recurrent networks. We implemented neuron-like agent
> interactions in which mutually visible agents operate as if they were
> reciprocally connected place cells in an attractor network. We attributed a
> phase state to agents to enable patterns of oscillatory synchronization
> similar to hippocampal models of theta-rhythmic (5–12 Hz) sequence
> generation. We demonstrate that multi-agent swarming and reward-approach
> dynamics can be expressed as a mobile form of Hebbian learning and that
> NeuroSwarms supports a single-entity paradigm that directly informs
> theoretical models of animal cognition. We present emergent behaviors
> including phase-organized rings and trajectory sequences that interact with
> environmental cues and geometry in large, fragmented mazes. Thus, NeuroSwarms
> is a model artificial spatial system that integrates autonomous control and
> theoretical neuroscience to potentially uncover common principles to advance
> both domains.

**Installation**

First, set up a new python environment (with either venv or Anaconda) and
install the required dependencies. Using Anaconda, you can enter these commands
in your shell:

```bash
$ conda create -n neuroswarms python ipython numpy scipy matplotlib pytables pillow
$ conda activate neuroswarms
```

Then, in the top-level `neuroswarms` folder, you can do a developer install
(with the `-e` option below) if you are interested in working with the code:

```bash
(neuroswarms)$ cd /path/to/neuroswarms
(neuroswarms)$ pip install -e .
```

If you have the `mpv` video player installed, it will be used to automatically
play the movie file of the simulation once it is saved. You can install `mpv`
with `brew install mpv` (macOS) or `sudo apt install mpv` (linux).

**Usage**

An example script in `scripts/run-example.py` shows how to create a
`NeuroswarmsModel` object and call its `.simulate(...)` method with parameter
values to run a trial simulation.

The `mapdata` folder contains the precomputed data for the two environments
presented in the paper: the 'multi-reward arena' and the 'large hairpin maze'.
You can choose either of these environments by setting the `env` parameter to
"test" or "hairpin", respectively.
