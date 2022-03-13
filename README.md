# AdaRL-code

Implementation codes and datasets used in paper **AdaRL: What, Where, and How to Adapt in Transfer Reinforcement Learning (ICLR'22 Spotlight)**.

---
Paper:
[[ICLR2022]](https://openreview.net/pdf?id=8H5bpVwvt5),
[[arXiv]](https://arxiv.org/abs/2107.02729)



## Introduction

One practical challenge in reinforcement learning (RL) is how to make quick adaptations when faced with new environments. In this paper, we propose a principled framework for adaptive RL, called AdaRL, that adapts reliably and efficiently to changes across domains with a few samples from the target domain, even in partially observable environments. Specifically, we leverage a parsimonious graphical representation that characterizes structural relationships over variables in the RL system. Such graphical representations provide a compact way to encode what and where the changes across domains are, and furthermore inform us with a minimal set of changes that one has to consider for the purpose of policy adaptation. We show that by explicitly leveraging this compact representation to encode changes, we can efficiently adapt the policy to the target domain, in which only a few samples are needed and further policy optimization is avoided. We illustrate the efficacy of AdaRL through a series of experiments that vary factors in the observation, transition and reward functions for Cartpole and Atari games.

## Citation

If you find our work helpful to your research, please consider citing our paper:

```
@InProceedings{huang2021adarl,
  title={AdaRL: What, Where, and How to Adapt in Transfer Reinforcement Learning},
  author={Huang, Biwei and Feng, Fan and Lu, Chaochao and Magliacane, Sara and Zhang, Kun},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022}
}
```


---

**Full information will be updated before/upon ICLR'22 conference starts.**

