# \lambda DNN: Achieving Predictable Distributed DNN Training with Serverless Architectures
\lambda DNN is a cost-efficient function resource provisioning framework to minimize the monetary cost and guarantee the performance for DDNN training workloads in serverless platforms.
## Overview of \lambda DNN
\lambda DNN framework running on AWS Lambda and comprises two pieces of modules: a training performance predictor and a function resource provisioner.To guarantee the objective DDNN training time, the resource provisioner further identifies the cost-efficient serverless function resource provisioning plan. Once the cost-efficient resource provisioning plan is determined, the function allocator finally sets up a number of functions with
an appropriate amount of memory.
<div align=center><img width="550" height="200" src="https://github.com/icloud-ecnu/lambdadnn/blob/master/images/implementation.png"/></div>
