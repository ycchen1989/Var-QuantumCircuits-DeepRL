# Variational-Quantum-Circuits-DeepReinforcementLearning

This is the numerical code for the article: \
[Variational Quantum Circuits for Deep Reinforcement Learning](https://arxiv.org/abs/1907.00397) (first released in Aug. 2019)


- This work awarded the Xanadu AI software competition 2019 research track first prize. [News](https://medium.com/xanaduai/xanadu-software-competition-the-results-are-in-9ccb6a3b591b?source=collection_home---6------5-----------------------)

We also sincerely thank the supports from Xanadu AI to provide PennyLane, which is a great value to the quantum AI community. 

## Requirements

with pip/conda install

```
pennylane
pytorch
matplotlib
qiskit
gym
numpy
```
## Run Code

- OpenAI Frozen Lake (please also refer to the [parameters](https://github.com/ycchen1989/Var-QuantumCircuits-DeepRL/blob/master/Code/ShortestPathFrozenLake.py#L4) )

```bash
python Code/QML_DQN_FROZEN_LAKE.py
```

- Cognitive Radio Game (network-simulator 3 style)

```
python Code/QML_DQN_NS3.py
```

## References 
If you find this work helps your research or use the code, please consider to cite our official reference. Thank you.

```
@article{chen2020variational,
  title={Variational quantum circuits for deep reinforcement learning},
  author={Chen, Samuel Yen-Chi and Yang, Chao-Han Huck and Qi, Jun and Chen, Pin-Yu and Ma, Xiaoli and Goan, Hsi-Sheng},
  journal={IEEE Access},
  year={2020},
  publisher={IEEE}
}
```
