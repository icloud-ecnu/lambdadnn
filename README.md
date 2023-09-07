# λDNN: Achieving Predictable Distributed DNN Training with Serverless Architectures
λDNN is a cost-efficient function resource provisioning framework to minimize the monetary cost and guarantee the performance for DDNN training workloads in serverless platforms.
## Overview of λDNN
λDNN framework running on AWS Lambda and comprises two pieces of modules: a training performance predictor and a function resource provisioner. To guarantee the objective DDNN training time, the resource provisioner further identifies the cost-efficient serverless function resource provisioning plan. Once the cost-efficient resource provisioning plan is determined, the function allocator finally sets up a number of functions with
an appropriate amount of memory.
<div align=center><img width="550" height="250" src="https://github.com/icloud-ecnu/lambdadnn/blob/master/images/implementation.png"/></div>

## Modeling DDNN Training Performance In Serverless Platforms
In general, the DNN model requires a number of iterations (denoted by k) to converge to an objective training loss value. Accordingly, the DDNN training
time T can be calculated by summing up the loading time, and the computation time, as well as the communication time, which is given by
<div align=center><img width="200" height="30" src="https://github.com/icloud-ecnu/lambdadnn/blob/master/images/eq-T.png"/></div>
The loading time is calculated as
<div align=center><img width="120" height="50" src="https://github.com/icloud-ecnu/lambdadnn/blob/master/images/eq-Tload.png"/></div>
Given n provisioned functions, the computation time tcomp of model gradients is defined as
<div align=center><img width="140" height="50" src="https://github.com/icloud-ecnu/lambdadnn/blob/master/images/eq-Tcomp.png"/></div>
The data communication time is calculated as
<div align=center><img width="120" height="50" src="https://github.com/icloud-ecnu/lambdadnn/blob/master/images/eq-Tcomm.png"/></div>
The objective is to minimize the monetary cost of provisioned function resources, while guaranteeing the performance of DDNN training
workloads. The optimization problem is formally defined as
<div align=center><img width="300" height="100" src="https://github.com/icloud-ecnu/lambdadnn/blob/master/images/eq-C.png"/></div>

## Publication
Fei Xu, Yiling Qin, Li Chen, Zhi Zhou, Fangming Liu, “[λDNN: Achieving Predictable Distributed DNN Training with Serverless Architectures](https://ieeexplore.ieee.org/document/9336272),” IEEE Transactions on Computers, 2022, 71(2): 450-463. DOI:10.1109/TC.2021.3054656.

```
@article{xu2021lambdadnn,
  title={$\lambda$dnn: Achieving predictable distributed DNN training with serverless architectures},
  author={Xu, Fei and Qin, Yiling and Chen, Li and Zhou, Zhi and Liu, Fangming},
  journal={IEEE Transactions on Computers},
  volume={71},
  number={2},
  pages={450--463},
  year={2021},
  publisher={IEEE}
}
```
