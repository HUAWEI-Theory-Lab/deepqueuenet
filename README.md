# DeepQueueNet: Towards Scalable and Generalized Network Performance Estimation with Packet-level Visibility
---

Demo paper in ACM SIGCOMM 2022


#### Abstract 
---
Network simulators are essential tools for network operators, and can assist important tasks such as capacity planning, topology design, and parameter tuning. Popular simulators are all based on discrete event simulation, and their performance does not scale with the size of modern networks. Recently, deep-learning-based techniques are introduced to solve the scalability problem, but, as we show with experiments, they have poor visibility in their simulation results, and cannot generalize to diverse scenarios. In this work, we combine scalable and generalized continuous simulation techniques with discrete event simulation to achieve high scalability, while achieving packet-level visibility. We start from a solid queueing-theoretic modelling of modern networks, and carefully identify the mathematically intractable or computationally-expensive parts, only which are then modelled using deep neural networks. Dubbed DeepQueueNet, our approach combines prior knowledge of networks, and can support arbitrary topology and device traffic management mechanisms (given sufficient training data). Our extensive experiments have shown that DeepQueueNet can achieve near-linear speed-up with the number of GPUs, and its accuracy of performance estimation is above 97.9% for average round-trip time (RTT) and 93.7% for 99th percentile RTT in all scenarios.


#### Description
---
In this demo, we illustrate how to use DeepQueueNet step-by-step to make end-to-end inference on a fattree topology with different traffic patterns (Traffic Pattern Generality):
- MAP
- Poisson
- On-off. 

The code of DeepQueueNet   used in the demo is  written in `tensorflow1.13.1` and  available on the [code_deepQueueNet](./code_deepQueueNet) directory.   Please check your environment before you run the following notebooks.  The first jupyter notebook (1. device model.ipynb) shows how to use device-level data to train a device model.  Namely, to predict the delay distribution of a packet given the past T time steps' information when it arrives at a N-port switch:

$$
[x_{t-T+1}, x_{t-T+1}, \cdots, x_t]\stackrel{f(\cdot)}{\longrightarrow} delay.
$$

<div align="center">
<img src="./assets/nodearch.png" alt="device_node"  width="250" height="230">
</div>

All the datasets used in the first step are available on the `data/4-port switch`  directory. 

After that, we present a demo to illustrate how to use the trained device model, deepQueueNet, to make end-to-end inference on a fattree topology. The task is deployed on 4 GPUs. 


<div align="center">
<img src="./assets/deploy.png" alt="deplo"  width="500" height="230">
</div>

All the datasets used in the second step are available on the `data/fattree16`  directory. 




#### `How to` guide
---
First of all, you can prepare the Python environment for the demo executing the following command:
```python
pip install -r requirements.txt
```
This command installs all the dependencies needed to execute DeepQueueNet.

Secondly, run `1. device model.ipynb`  to get the trained device model.  To train and evaluate DeepQueueNet,  we provide some ready-to-use datasets. They are generated by a MAP generator and  simulated with a packet-level simulator (ns.py). The trained model is saved in the `trained` folder.  Interested readers can collect some real-world traces or generate some synthetic traces from other generators to train the model.  How many traces to use depends on your requirements on its accuracy. Due to space limitations of GitHub LFS, we host the content of `data` and `trained` [on Dropbox](https://www.dropbox.com/s/q56sx4hxe93n4g5/DeepQueueNet-dataset.zip?dl=0) instead. Please download the zipped archive and extract the two folders named `data` and `trained` both to root directory before proceeding.

As an example, we provide below two figures that show the evolution of the loss (Mean Absolute Error) and the accuracy (Pearson correlation coefficient) of the device model that is provided in the [trained](./trained) directory.

1. learning curve

<div align="center">
<img src="./assets/learning_curve.png" alt="learning_curve"  width="500" height="150">
</div>

2. regression rho

<div align="center">
<img src="./assets/regression.png" alt="rho"  width="500" height="160">
</div>
3. distribution

<div align="center">
<img src="./assets/train.png" alt="dist_train"  width="500" height="160">
</div>
<div align="center">
<img src="./assets/test_endogeny.png" alt="dist_test1"  width="500" height="160">
</div>
<div align="center">
<img src="./assets/test_exogenesis.png" alt="dist_test2"  width="500" height="160">
</div>

Finally, run `2. traffic gen. on fattree16.ipynb.` to use the trained device model, deepQueueNet, to make end-to-end inference on fattree topology.


#### Licensing
---
Please see [LICENSE](./LICENSE.txt) for more info.
