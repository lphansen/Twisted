#!/bin/bash
#SBATCH --job-name=two_capital
#SBATCH --output=two_capital.out
#SBATCH --error=two_capital.err
#SBATCH --time=4:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28


cd $HOME/Desktop/Twisted

julia ./newsets_twocapitals.jl