![Employee data](/doc/animation.gif?raw=true "Employee Data title")
# Bayesion Optimization for a Pendulum
Actively learning the disturbance of a dynamical system. The disturbance is modeled with a PD controller

## Setup
```
pip install virtualenv --user
virtualenv venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Run
To train the model run

```
 python main.py train
```
To show plot the last state run
```
 python main.py plot-end
```
To plot the animation run
```
 python main.py plot
```
To create a gif run
```
 python main.py make-gif
```

```
 python main.py --help
```
Most settings can be changed in `config.txt`
