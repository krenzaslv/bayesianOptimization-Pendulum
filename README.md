![Employee data](/doc/animation.gif?raw=true "Employee Data title")
# Bayesian Optimization for disturbed dynamics
Actively learning the disturbance of a dynamical system. The disturbance is modeled with a PD controller

## Setup
```
pip install virtualenv --user
virtualenv venv
. .venv/bin/activate
pip install -r requirements.txt
```
## Settings
Most settings can be changed in `config.txt`

## Run
To train the model run

```
 python main.py train
```
## Visualize
To visualize various metrics run
```
tensorboard --logdir=runs

```
To visualize the search process run
```
 python main.py plot-end
 python main.py plot
```
To create a gif after the plot run
```
 python main.py make-gif
```
## Help

```
 python main.py --help
```
