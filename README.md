# QMDP-net

Implementation of the NIPS 2017 paper: 

QMDP-Net: Deep Learning for Planning under Partial Observability  
Peter Karkus, David Hsu, Wee Sun Lee  
National University of Singapore  
https://arxiv.org/abs/1703.06692

The code implements the 2D grid navigation domain, and a QMDP-net with 2D state space in tensorflow.

### Requirements

Python 2.7  
Tensorflow 1.3.0  
Python packages: numpy, scipy, tables, pymdptoolbox, tensorpack

To install these packages using pip:
```
pip install tensorflow
pip install numpy scipy tables pymdptoolbox tensorpack
```

Optional: to speed up data generation download and install the latest version of pymdptoolbox
```
git clone https://github.com/sawcordwell/pymdptoolbox.git pymdptoolbox
cd ./pymdptoolbox
python setup.py install
```


### Train and evaluate a QMDP-net

The folder ./data/grid10 contains training and test data for the deterministic 10x10 grid navigation domain
(10,000 environments, 5 trajectories each for training, 500 environments, 1 trajectory each for testing).


Train network using only the first 4 steps of each training trajectory:
```
python train.py ./data/grid10/ --logpath ./data/grid10/output-lim4/ --lim_traj_len 4
```
The learned model will be saved to ./data/grid10/output-lim4/final.chk
 

Load the previously saved model and train further using the full trajectories:
```
python train.py ./data/grid10/ --logpath ./data/grid10/output-lim100/ --loadmodel ./data/grid10/output-lim4/final.chk --lim_traj_len 100
```


For help on arguments execute:
```
python train.py --help
```


### Generate data

Generate data for the 18x18 deterministic grid navigation domain.  
10,000 environments for training, 500 for testing, 5 and 1 trajectories per environment

```
python grid.py ./data/grid18/ 10000 500 --N 18 --train_trajs 5 --test_trajs 1
```


For help on arguments execute:
```
python grid.py --help
```

