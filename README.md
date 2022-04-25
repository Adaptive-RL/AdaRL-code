# AdaRL

Implementation codes and datasets used in paper **AdaRL: What, Where, and How to Adapt in Transfer Reinforcement Learning (ICLR'22 Spotlight)**.

---
Paper:
[[ICLR2022]](https://openreview.net/pdf?id=8H5bpVwvt5),
[[arXiv]](https://arxiv.org/abs/2107.02729)



## Introduction

One practical challenge in reinforcement learning (RL) is how to make quick adaptations when faced with new environments. In this paper, we propose a principled framework for adaptive RL, called AdaRL, that adapts reliably and efficiently to changes across domains with a few samples from the target domain, even in partially observable environments. Specifically, we leverage a parsimonious graphical representation that characterizes structural relationships over variables in the RL system. Such graphical representations provide a compact way to encode what and where the changes across domains are, and furthermore inform us with a minimal set of changes that one has to consider for the purpose of policy adaptation. We show that by explicitly leveraging this compact representation to encode changes, we can efficiently adapt the policy to the target domain, in which only a few samples are needed and further policy optimization is avoided. We illustrate the efficacy of AdaRL through a series of experiments that vary factors in the observation, transition and reward functions for Cartpole and Atari games.


## Requirements
The current version of the code has been tested with following libs:
* `cudatoolkit==10.0.130`
* `tensorflow-gpu 1.15.0`
* `cudatoolkit 10.0.130`
* `gym 0.18.0`
* `Pillow`
* `keras 2.6.0`
* `numpy 1.20.2`
* `opencv-python 4.5.1.48`

Install the required the packages inside the virtual environment:
```
$ conda create -n yourenvname python=3.7 anaconda
$ source activate yourenvname
$ conda install cudatoolkit==10.0.130
$ pip install -r requirements.txt
```
### Installation for envs 
#### Cartpole
```
cd libs/gym-cartpole-world-master
pip install -e .
```
Take gravity of 9.8 as an example.
```
python gym_cartpole_world_usage.py
```
All the version details can be found in ```libs/gym-cartpole-world-master/gym_cartpole_world/__init__.py```.
##### Data preparation
Take cartpole with gravity of 5 as an example.
```
cd ../../
python data/data_gen_cartpoleworld.py 'v00' 
```
##### Atari Pong
Install ```gym_pong```.
  ```
  cd libs/gym_pong-master
  pip install -e .
  ```

Follow [this](https://github.com/openai/atari-py#roms) instruction to import ROMs

Take size of 2.0 as an example.
  ```
  python gym_pong_usage.py
  ```

  All the version details can be found in ```libs/gym_pong-master/gym_pong/__init__.py```.

##### Data preparation
Take the domain in size of 2.0 as an example
```
cd ../../
python data/data_gen_pong.py -name 'Pongm' -v v00
```
For the reward-varying cases
```
python data/data_gen_pong.py -name 'Pong' -v v0 -rewardm 'linear' -k1 0.1

python data/data_gen_pong.py -name 'Pong' -v v0 -rewardm 'non-linear' -k2 2.0
```
## Model estimation
Take cartpole with domain across gravities as an example
```
python model_est.py -name cartpole -source ./dataset/Cartpole/G_len40_xthreshold5_thetathreshold45_trial10000 -dest ./dataset/Cartpole/G_len40_xthreshold5_thetathreshold45_trial10000/v_train -domain 00 01 02 03 04
```
P.S. instead of directly training the whole model, we divide the whole process into 4 steps, i.e., VAE, series, dynamics and then the whole model with parameters initialized by previous parts.

## Policy optimization
Take cartpole with domain across gravities as an example
```
python policy_opt.py -name cartpole  -mvae_p ./results/Cartpole/G_len40_xthreshold5_thetathreshold45_trial10000/all/{TIME_NOW}/all.json -source ./dataset/Cartpole/G_len40_xthreshold5_thetathreshold45_trial10000 -domain 00 01 02 03 04
```

## On testing data 
```
python test.py -name cartpole -source ./dataset/Cartpole/G_len40_xthreshold5_thetathreshold45_trial10000 -dest ./dataset/Cartpole/G_len40_xthreshold5_thetathreshold45_trial10000/v_test -domain 05 06 -mvae_p ./results/Cartpole/G_len40_xthreshold5_thetathreshold45_trial10000/all/{TIME_NOW}/all.json -k 0 -step 1
python test.py -name cartpole -source ./dataset/Cartpole/G_len40_xthreshold5_thetathreshold45_trial10000 -dest ./dataset/Cartpole/G_len40_xthreshold5_thetathreshold45_trial10000/v_test -domain 05 06 -mvae_p ./results/Cartpole/G_len40_xthreshold5_thetathreshold45_trial10000/all/{TIME_NOW}/all.json -k 0 -step 2
```

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

## Acknowledgements
Parts of code were built upon [world model implementation by David Ha](https://github.com/hardmaru/WorldModelsExperiments).