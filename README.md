# 3DTicTacToe
Group Project for COMP 560

Completed with Andrew Jacober and Sarah Hand

Using Q-Learning to train an AI to play 4x4x4 Tic Tac Toe

The python file consists of 2 objects, State and Player, that use Q-Learning to train an AI agent to play 4x4x4 tic tac toe.

Make sure the following python libraries are installed on your machine before running this code: pandas numpy csv

The code should be run in the following format in a console:

python 560_HW2.py x y z

where x, y, and z are some integer value for how many games the AI should play against itself

Example:

python 560_HW2.py 500 1000 2000

INPUTS:

There is a CSV file that is required to run the code as well

This file must be named "start_states.csv"

This CSV file works as a helper to some of the panda dataframe functions to output the learned Q-Values for each of the 64 starting states of the tic tac toe game.

This was done to make it easier to read the resulting values rather than have them all output in the console

OUTPUTS:

The code will output 6 CSV files:

One full q-value table for each of the 3 training values (x, y and z)

One partial q-value table of just the 64 possible starting positions for each of the 3 training values (x, y, and z)
