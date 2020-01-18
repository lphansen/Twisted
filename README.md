# Twisted Probabilities, Uncerntainty, and Prices
This repository contains codes and jupyter notebooks which estimates and demonstrates results of the empirical model in "Twisted Probabilities, Uncerntainty, and Prices" by [Lars Peter Hansen][id1], [Bálint Szőke][id2] and Lloyd S. Han and [Thomas J. Sargent][id3]. Latest version could be found [here][id4].

[id1]: https://larspeterhansen.org/
[id2]: https://www.balintszoke.com/
[id3]: http://www.tomsargent.com/
[id4]: https://larspeterhansen.org/research/papers/

## To-dos:
1. Ask for feedbacks from Tom and Lars

## File structures
    - main.ipynb is a notebook producing interactive figures accompanying our paper
    - single_capital.ipynb illustrate and demonstrate our code for solving our model in single capital stock case
    - two_capitals.ipynb illustrate and demonstrate our code for solving our model in two capital stocks case
Binder: 

## Acessing our jupyter notebook
To access our notebook, please follow steps below:
1.	Open a Windows command prompt or Mac command terminal and change into the folder you would like to store the files. 
    - You can do this using the command __cd__ in the command prompt.    
    - For example, running "cd “C:\Users\username\python” " (don’t forget “” around the path name to use an absolute path) would lead me to my designated folder.
```
cd [folder path name]
```
2.	Clone the github repository for the paper 
    - If you don’t have github installed, try installing it from this page: https://git-scm.com/download/win.
    - You can do this by running in the command prompt. 
```
git clone https://github.com/lphansen/Twisted
```
3.  If User don't have Julia, install Julia and add Julia executabe to system environment paths.
    - For installing Julia, please visit https://julialang.org/
    - a) Mac user: 
        - i) type in terminal: PATH="/Applications/Julia-1.3.app/Contents/Resources/julia/bin/:${PATH}"
        - ii) type in terminal: export PATH
        - Note: remember to change the version of Julia in the path of i) if your Julia version is not 1.3 
    - b) Windows user:
        - follow instructions here: http://wallyxie.com/weblog/adding-julia-windows-path-command-prompt/
        - or you can visit https://en.wikibooks.org/wiki/Introducing_Julia/Getting_started
4.	Go back to command line prompt, change directories into the __Twisted__ folder and open jupyter notebook by running below in command prompt
    - If you don’t have anaconda3 and jupyter notebook installed, try installing from this page: https://jupyter.org/install.html
```
jupyter notebook
```
5. Our notebook interface should show up in the browser and have fun playing with our notebooks!


