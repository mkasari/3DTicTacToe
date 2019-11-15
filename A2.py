import numpy as np
import csv
import time
import pandas as pd
import sys

BOARD_ROWS = 8
BOARD_COLS = 8


class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS)).astype(int)
        self.p1 = p1
        self.p2 = p2
        self.gameOver = False
        self.boardHash = None
        #p1 plays first
        self.playerSymbol = 1

    #hash board to 1d for easy storage
    def getHash(self):
        self.boardHash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        return self.boardHash

    #determining winner and returning user winner value (-1 or 1)
    #will also return 0 for a tie and None if the game is not over yet
    def winner(self):
        #ROWS
        for i in range(BOARD_ROWS):
            #first layer of board
            if i//2 == 0:
                if sum(self.board[i][0:4])==4:
                    self.gameOver = True
                    return 1
                if sum(self.board[i][4:8])==4:
                    self.gameOver = True
                    return 1

                if sum(self.board[i][0:4])==-4:
                    self.gameOver = True
                    return -1
                if sum(self.board[i][4:8])==-4:
                    self.gameOver = True
                    return -1

            #second layer of board
            elif i//2 == 1:
                if sum(self.board[i][0:4])==4:
                    self.gameOver = True
                    return 1
                if sum(self.board[i][4:8])==4:
                    self.gameOver = True
                    return 1

                if sum(self.board[i][0:4])==-4:
                    self.gameOver = True
                    return -1
                if sum(self.board[i][4:8])==-4:
                    self.gameOver = True
                    return -1

            #third layer of board
            elif i//2 == 2:
                if sum(self.board[i][0:4])==4:
                    self.gameOver = True
                    return 1
                if sum(self.board[i][4:8])==4:
                    self.gameOver = True
                    return 1

                if sum(self.board[i][0:4])==-4:
                    self.gameOver = True
                    return -1
                if sum(self.board[i][4:8])==-4:
                    self.gameOver = True
                    return -1

            #fourth layer of board
            elif i//2 == 3:
                if sum(self.board[i][0:4])==4:
                    self.gameOver = True
                    return 1
                if sum(self.board[i][4:8])==4:
                    self.gameOver = True
                    return 1
                if sum(self.board[i][0:4])==-4:
                    self.gameOver = True
                    return -1
                if sum(self.board[i][4:8])==-4:
                    self.gameOver = True
                    return -1


        #COLS
        for i in range(0,8,2):
            for j in range(4):
                if self.board[i][j] + self.board[i][j+4] + self.board[i+1][j] + self.board[i+1][j+4] == 4:
                    self.gameOver = True
                    return 1

                if self.board[i][j] + self.board[i][j+4] + self.board[i+1][j] + self.board[i+1][j+4] == -4:
                    self.gameOver = True
                    return -1


        #DIAG
        #1
        if self.board[0][0] + self.board[0][5] + self.board[1][2] + self.board[1][7] == 4:
            self.gameOver = True
            return 1
        if self.board[0][0] + self.board[0][5] + self.board[1][2] + self.board[1][7] == -4:
            self.gameOver = True
            return -1

        if self.board[0][3] + self.board[0][6] + self.board[1][1] + self.board[1][4] == 4:
            self.gameOver = True
            return 1
        if self.board[0][3] + self.board[0][6] + self.board[1][1] + self.board[1][4] == -4:
            self.gameOver = True
            return -1
        #2
        if self.board[2][0] + self.board[2][5] + self.board[3][2] + self.board[3][7] == 4:
            self.gameOver = True
            return 1
        if self.board[2][0] + self.board[2][5] + self.board[3][2] + self.board[3][7] == -4:
            self.gameOver = True
            return -1

        if self.board[2][3] + self.board[2][6] + self.board[3][1] + self.board[3][4] == 4:
            self.gameOver = True
            return 1
        if self.board[2][3] + self.board[2][6] + self.board[3][1] + self.board[3][4] == -4:
            self.gameOver = True
            return -1
        #3
        if self.board[4][0] + self.board[4][5] + self.board[5][2] + self.board[5][7] == 4:
            self.gameOver = True
            return 1
        if self.board[4][0] + self.board[4][5] + self.board[5][2] + self.board[5][7] == -4:
            self.gameOver = True
            return -1

        if self.board[4][3] + self.board[4][6] + self.board[5][1] + self.board[5][4] == 4:
            self.gameOver = True
            return 1
        if self.board[4][3] + self.board[4][6] + self.board[5][1] + self.board[5][4] == -4:
            self.gameOver = True
            return -1
        #4
        if self.board[6][0] + self.board[6][5] + self.board[7][2] + self.board[7][7] == 4:
            self.gameOver = True
            return 1
        if self.board[6][0] + self.board[6][5] + self.board[7][2] + self.board[7][7] == -4:
            self.gameOver = True
            return -1

        if self.board[6][3] + self.board[6][6] + self.board[7][1] + self.board[7][4] == 4:
            self.gameOver = True
            return 1
        if self.board[6][3] + self.board[6][6] + self.board[7][1] + self.board[7][4] == -4:
            self.gameOver = True
            return -1

        #DEPTH
        for j in range(8):
            if (self.board[0][j] + self.board[2][j] + self.board[4][j] + self.board[6][j] == 4 or
            self.board[1][j] + self.board[3][j] + self.board[5][j] + self.board[7][j] == 4):
                self.gameOver = True
                return 1

            elif (self.board[0][j] + self.board[2][j] + self.board[4][j] + self.board[6][j] == -4 or
            self.board[1][j] + self.board[3][j] + self.board[5][j] + self.board[7][j] == -4):
                self.gameOver = True
                return -1

        #DEPTH DIAG
        if self.board[0][0] + self.board[2][5] + self.board[5][2] + self.board[7][7] == 4:
            self.gameOver = True
            return 1
        if self.board[0][0] + self.board[2][5] + self.board[5][2] + self.board[7][7] == -4:
            self.gameOver = True
            return -1

        if self.board[0][3] + self.board[2][6] + self.board[5][1] + self.board[7][4] == 4:
            self.gameOver = True
            return 1
        if self.board[0][3] + self.board[2][6] + self.board[5][1] + self.board[7][4] == -4:
            self.gameOver = True
            return -1

        if self.board[1][4] + self.board[3][1] + self.board[4][6] + self.board[6][3] == 4:
            self.gameOver = True
            return 1
        if self.board[1][4] + self.board[3][1] + self.board[4][6] + self.board[6][3] == -4:
            self.gameOver = True
            return -1

        if self.board[1][7] + self.board[3][2] + self.board[4][5] + self.board[6][0] == 4:
            self.gameOver = True
            return 1
        if self.board[1][7] + self.board[3][2] + self.board[4][5] + self.board[6][0] == -4:
            self.gameOver = True
            return -1


        # tie
        # no available positions
        if len(self.openPositions()) == 0:
            self.gameOver = True
            return 0

        # game not over
        self.gameOver = False
        return None

    #finds each available position on the board and returns an array
    #   of those positions
    def openPositions(self):
        positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                if self.board[i, j] == 0:
                    positions.append((i, j))  # need to be tuple
        return positions

    #adds the players symbol to the board to update the state
    def updateBoardState(self, position):
        self.board[position] = self.playerSymbol
        # switch to another player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    #only when game ends
    #pushes the win, loss, or tie value to the feedReward function
    #   depending on the winner of the game
    def addReward(self):
        result = self.winner()
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(-1)
        else:
            self.p1.feedReward(0.05)
            self.p2.feedReward(0.5)

    #resets the board to its initial start state to prep for another round
    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS)).astype(int)
        self.boardHash = None
        self.gameOver = False
        self.playerSymbol = 1

    #function that puts together the other functions to actually play the game
    def train(self, rounds):
        for i in range(rounds):

            # if number of rounds is less than 10, then show every round
            if rounds < 10:
                j=i+1
                print("Round {} of {}".format(j, rounds))

            # if number of rounds is less than 500, then show every 50 rounds
            if rounds < 500:
                if (i == 0 or i%50 == 49):
                    j = i+1
                    print("Round {} of {}".format(j, rounds))

            # if number of rounds is between 500 and 100, then show every 100 rounds
            elif rounds >=500 and rounds <1000:
                if (i == 0 or i%100 == 99):
                    j = i+1
                    print("Round {} of {}".format(j, rounds))

            # if number of rounds is greater than 1000, then show every 500 rounds
            elif rounds >=1000:
                if (i == 0 or i%500 == 499):
                    j = i+1
                    print("Round {} of {}".format(j, rounds))
            while not self.gameOver:
                #Player 1
                positions = self.openPositions()
                p1_move = self.p1.makeMove(positions, self.board, self.playerSymbol)
                #take action and upate board state
                self.updateBoardState(p1_move)
                board_hash = self.getHash()
                self.p1.addState(board_hash)

                #checks if the game has a winner yet
                win = self.winner()

                #if there is a winner reset the objects
                if win is not None:
                    #ended with p1 either win or draw
                    self.addReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break

                else:
                    # Player 2
                    positions = self.openPositions()
                    p2_move = self.p2.makeMove(positions, self.board, self.playerSymbol)
                    #take action and update board state
                    self.updateBoardState(p2_move)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    #checks if there is a winner now that p2 has made a play
                    win = self.winner()

                    #if there is a winner reset the objects
                    if win is not None:
                        #ended with p2 either win or draw
                        self.addReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break


    #This is an alternate play function that is only used when AI is playing
    #   against a human player
    #Works similarly to other play function but has a different p2 win scenario
    def test(self):
        while not self.gameOver:
            # Player 1
            positions = self.openPositions()
            p1_move = self.p1.makeMove(positions, self.board, self.playerSymbol)
            self.updateBoardState(p1_move)
            print(self.board)


            #check board status if it is end
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(self.p1.name, "wins!")
                else:
                    print("tie!")
                self.reset()
                break

            else:
                # Player 2
                positions = self.openPositions()
                p2_move = self.p2.makeMove(positions)
                self.updateBoardState(p2_move)
                print(self.board)
                win = self.winner()
                if win is not None:
                    if win == -1:
                        print(self.p2.name, "wins!")
                    else:
                        print("tie!")
                    self.reset()
                    break


class Player:

    #initialize AI player
    def __init__(self, name):
        self.name = name
        self.states = []
        self.lr = 0.4
        self.exp_rate = 0.2
        self.discount_factor = 0.9
        self.states_value = {}

    #board hash function for the player object to call
    #works the same as the State board hash function
    def getHash(self, board):
        boardHash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return boardHash

    #function that chooses the move for the AI based on the learned values
    #   or explored if a random number is below the explore rate value
    def makeMove(self, positions, current_board, symbol):
        #EXPLORE
        if np.random.uniform(0, 1) <= self.exp_rate:
            # take random action if below explore rate value
            idx = np.random.choice(len(positions))
            action = positions[idx]
        #EXPLOIT
        else:
            value_max = -999
            count = 0
            actions = []
            #checks each next possible move from current state and makes a
            #   decision based on the learned q-value table
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_boardHash = self.getHash(next_board)
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
                value = float(value)
                value_max = float(value_max)
                if value >= value_max:
                    value_max = value
                    count += 1
                    actions.append(p)
                    if count > 1:
                        randn_even_action = np.random.choice(len(actions))
                        action = actions[randn_even_action]
                    else:
                        action = p
        return action

    #append a hash state to the array to add to q-table later
    def addState(self, state):
        self.states.append(state)

    #at the end of game, backpropagate and update states value (q-table)
    #this is where the q-values are calculated
    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += round((self.lr * (reward + (self.discount_factor * self.states_value[st]))), 3)
            reward = self.states_value[st]

    #resets the player objects to play a new game
    def reset(self):
        self.states = []

    #saves the policy so it can be loaded later in an AI vs Human game
    def savePolicy(self, rounds):
        w = csv.writer(open("policy_" + str(self.name) + "-" + str(rounds) + "rounds.csv", "w"))
        w.writerow(["state", "q-value"])
        for key, val in sorted(self.states_value.items(), reverse=True):
            w.writerow([key, val])
        print("policy saved for:", self.name, " - ", rounds, " rounds")

    #loads policy for AI player to read while playing against a human
    def loadPolicy(self, file):
        with open(file, mode='r') as infile:
            reader = csv.reader(infile)
            self.states_value = {rows[0]:rows[1] for rows in reader}

    #extracts the q-values for the starting 64 positions
    def getInitialQvals(self, policy_file, init_state_file):
        readPolicy = pd.read_csv(policy_file)
        readPolicy = readPolicy.replace(r'\n','', regex=True)
        readPolicy = readPolicy.replace(r'  ', ' ', regex=True)

        initialStates = pd.read_csv(init_state_file)

        initStateValues = initialStates.merge(readPolicy, on='state', how='left')

        initStateValues = initStateValues[['state', 'q-value_y']]

        initStateValues.rename(columns={'q-value_y':'q-value'})

        return initStateValues

    #saves the q-values for starting 64 positions to a csv file to view
    def saveInitialQvals(self, initStateValues, rounds):
        filename = 'initStateValues-' + str(rounds) + 'rounds.csv'
        initStateValues.to_csv(filename)





class HumanPlayer:

    #initialize human player
    def __init__(self, name):
        self.name = name

    #human inputs their move using row and column, and if valid, make the move
    def makeMove(self, positions):
        while True:
            row = int(input("Input your action row:"))
            col = int(input("Input your action col:"))
            action = (row, col)
            if action in positions:
                return action



if __name__ == "__main__":

    #timer for reference
    def time_elapsed(start, end):
        hr, rem = divmod(end-start, 3600)
        mins, sec = divmod(rem, 60)
        print("finished in " + "{:0>2}:{:0>2}:{:05.2f}".format(int(hr), int(mins), sec))

    #training
    p1 = Player("AI1")
    p2 = Player("AI2")


    rounds1 = int(sys.argv[1])
    rounds2 = int(sys.argv[2])
    rounds3 = int(sys.argv[3])

    st = State(p1, p2)
    print("training for: ", rounds1, " rounds")
    s1 = time.time()
    st.train(rounds1)
    e1 = time.time()
    time_elapsed(s1, e1)
    p1.savePolicy(rounds1)
    print("  ")

    st = State(p1, p2)
    print("training for: ", rounds2, " rounds")
    s2 = time.time()
    st.train(rounds2)
    e2 = time.time()
    time_elapsed(s2, e2)
    p1.savePolicy(rounds2)
    print("  ")

    st = State(p1, p2)
    print("training for: ", rounds3, " rounds")
    s3 = time.time()
    st.train(rounds3)
    e3 = time.time()
    time_elapsed(s3, e3)
    p1.savePolicy(rounds3)

    policy_file1 = 'policy_AI1-' + str(rounds1) + 'rounds.csv'
    policy_file2 = 'policy_AI1-' + str(rounds2) + 'rounds.csv'
    policy_file3 = 'policy_AI1-' + str(rounds3) + 'rounds.csv'
    init_state_file = 'start_states.csv'

    initial_state_values_1 = p1.getInitialQvals(policy_file1, init_state_file)
    initial_state_values_2 = p1.getInitialQvals(policy_file2, init_state_file)
    initial_state_values_3 = p1.getInitialQvals(policy_file3, init_state_file)

    p1.saveInitialQvals(initial_state_values_1, rounds1)
    p1.saveInitialQvals(initial_state_values_2, rounds2)
    p1.saveInitialQvals(initial_state_values_3, rounds3)
    print('saved starting q-values for all 3 training rounds')


    #Testing
    #Uncomment below before running to play an AI vs Human game
    #Be sure to write in the name of the policy file you want AI to use

#     p1 = Player("computer", exp_rate=0
#     policy_file = ""
#     p1.loadPolicy(policy file)

#     p2 = HumanPlayer("human")

#     st = State(p1, p2)
#     st.test()
