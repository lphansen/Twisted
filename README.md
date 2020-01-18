# Twisted Probabilities, Uncerntainty, and Prices
This repository contains codes and jupyter notebooks which estimates and demonstrates results of the empirical model in "Twisted Probabilities, Uncerntainty, and Prices" by [Lars Peter Hansen][id1], [Bálint Szőke][id2] and Lloyd S. Han and [Thomas J. Sargent][id3]. Latest version could be found [here][id4].

[id1]: https://larspeterhansen.org/
[id2]: https://www.balintszoke.com/
[id3]: http://www.tomsargent.com/
[id4]: https://larspeterhansen.org/research/papers/

## To-dos:
1. Ask for feedbacks from Tom and Lars

## File structures
1. __main.ipynb__ is a notebook producing interactive figures accompanying our paper
2. __single_capital.ipynb__ illustrate and demonstrate our code for solving our model in single capital stock case
3. __two_capitals.ipynb__ illustrate and demonstrate our code for solving our model in two capital stocks case
We have a Binder for users to play with our notebook without setting up files on their local machine: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lphansen/Twisted/master)

## Prerequisites

This project simply requires the Anaconda distribution of Python version 3.x and Julia 1.1.x. Additional dependencies and prerequisites are handled automatically in setup.

## Installing the environment and 

Navigate to the folder containing the code and set up the virtual environment necessarily to run our code

For Mac Users, please open the terminal and run the following commands in order
```
cd /path
git clone https://github.com/lphansen/Twisted.git
cd Twisted
source setup.sh
```
For Windows Users, please open command prompt (shortcut: Windows+R and type 'cmd'）
```
cd /path
git clone https://github.com/lphansen/Twisted.git
conda update conda
conda env create -f environment.yml
conda activate Twisted
cd Twisted
```
Please replace /path to user designated folder path in both cases.

Press `y` to proceed with installation when prompted. You will know that setup has been correctly implemented if the word `(tenuous)` contained in parenthesis appears on the current line of your terminal window.

## Jupyter Notebook for Interactive Plots in the Paper

To run the notebook, simply use: (Make sure acitivating our virtual python environment "Twisted" and navigating to this folder)
```
jupyter notebook
```

Our notebook interface should show up in the browser and have fun playing with our notebooks!


