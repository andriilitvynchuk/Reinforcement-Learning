from tqdm import tqdm
import numpy as np
LENGTH = 3

def play_game(p1, p2, env, draw=False):
    current_player = None

    while not env.game_over():
        if current_player == p1:
            current_player = p2
        else:
            current_player = p1

        if draw:
            if (draw==1 and current_player==p1) or (draw==2 and current_player==p2):
                env.draw_board()

        current_player.take_action(env)

        state = env.get_state()
        p1.update_state_history(state)
        p2.update_state_history(state)

        if draw:
            env.draw_board()

    p1.update(env)
    p2.update(env)


def get_state_hash_and_winner(env, i=0, j=0):
    results=[]

    for v in (0, env.x, env.o):
        env.board[i, j] = v
        if j == 2:
            if i == 2:
                state = env.get_state()
                ended = env.game_over(force_recalculate=True)
                winner = env.winner
                results.append((state, winner, ended))
            else:
                results += get_state_hash_and_winner(env, i + 1, 0)
        else:
            results += get_state_hash_and_winner(env, i, j + 1)

    return results


def initialV_x(env, state_winner_results):
    V = np.zeros(env.num_states)
    for state, winner, ended in state_winner_results:
        if ended:
            if winner==env.x:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v

    return V


def initialV_o(env, state_winner_results):
    V = np.zeros(env.num_states)
    for state, winner, ended in state_winner_results:
        if ended:
            if winner == env.o:
                v = 1
            else:
                v = 0
        else:
            v = 0.5
        V[state] = v

    return V


class Environment:

    def __init__(self):
        self.board = np.zeros((LENGTH, LENGTH))
        self.x = -1
        self.o = 1
        self.winner = None
        self.ended = False
        self.num_states = 3**(LENGTH * LENGTH)


    def is_empty(self, i, j):
        return self.board[i, j] == 0


    def reward(self, sym):
        if not self.game_over():
            return 0

        return 1 if self.winner == sym else 0


    def get_state(self):
        power = 0
        state = 0
        for i in range(LENGTH):
            for j in range(LENGTH):
                if self.board[i, j] == 0:
                    temp = 0
                elif self.board[i, j] == self.x:
                    temp = 1
                elif self.board[i, j] == self.o:
                    temp = 2
                state += (3 ** power) * temp
                power += 1

        return state


    def game_over(self, force_recalculate=False):
        if not force_recalculate and self.ended:
            return self.ended
        for i in range(LENGTH):
            for player in (self.x, self.o):
                if self.board[i].sum() == player * LENGTH or self.board[:, i].sum() == player * LENGTH or self.board.trace() == player * LENGTH or np.fliplr(self.board).trace() == player * LENGTH:
                    self.winner = player
                    self.ended = True
                    return True

        if np.all((self.board == 0) == False):
            self.winner = None
            self.ended = True
            return True

        self.winner = None
        return False

        return False


    def draw_board(self):
        for i in range(LENGTH):
            print('--------------')
            for j in range(LENGTH):
                if self.board[i, j] == self.x:
                    print('| x |', end = ' ')
                elif self.board[i, j] == self.o:
                    print('| o |', end = ' ')
                else:
                    print('|  |', end = ' ')
            print('\n')
        print('--------------')


class Agent:

    def __init__(self, eps=0.1, alpha=0.5):
        self.eps = eps
        self.alpha = alpha
        self.verbose = False
        self.state_history = []


    def setV(self, V):
        self.V = V


    def set_symbol(self, sym):
        self.sym = sym


    def set_verbose(self, v):
        self.verbose = v


    def reset_history(self):
        self. state_history = []


    def set_eps(self, eps):
        self.eps = eps

    def take_action(self, env):
        r = np.random.rand()
        best_state = None

        if r < self.eps:
            if self.verbose:
                print('Taking a random action')

            possible_moves = []
            for i in range(LENGTH):
                for j in range(LENGTH):
                    if env.is_empty(i, j):
                        possible_moves.append((i, j))
            idx = np.random.choice(len(possible_moves))
            next_move = possible_moves[idx]
        else:
            pos2value = {}
            next_move = None
            best_value = -1
            for i in range(LENGTH):
                for j in range(LENGTH):
                    if env.is_empty(i, j):
                        env.board[i, j] = self.sym
                        state = env.get_state()
                        env.board[i, j] = 0
                        pos2value[(i, j)] = self.V[state]
                        if self.V[state] > best_value:
                            best_value = self.V[state]
                            best_state = state
                            next_move = (i, j)
            if self.verbose:
                print('Take a greedy action')
        # print(next_move)
        # print(env.board)
        # print(env.ended)
        env.board[next_move[0], next_move[1]] = self.sym


    def update_state_history(self, state):
        self.state_history.append(state)


    def update(self, env):
        reward = env.reward(self.sym)
        target = reward
        for prev in reversed(self.state_history):
            value = self.V[prev] + self.alpha * (target - self.V[prev])
            self.V[prev] = value
            target = value
        self.reset_history()


class Human:
    def __init__(self):
        pass


    def set_symbol(self, sym):
        self.sym = sym


    def take_action(self, env):
        while True:
            move = input("Enter coordinates i, j for your next move (i, j = 0..2):")
            i, j = move.split(',')
            i, j = int(i), int(j)
            if env.is_empty(i, j):
                env.board[i, j] = self.sym
                break


    def update(self, env):
        pass


    def update_state_history(self, s):
        pass


if __name__ == '__main__':
    p1 = Agent()
    p2 = Agent()

    env = Environment()
    state_winner_triples = get_state_hash_and_winner(env)

    Vx = initialV_x(env, state_winner_triples)
    p1.setV(Vx)
    Vo = initialV_o(env, state_winner_triples)
    p2.setV(Vo)

    p1.set_symbol(env.x)
    p2.set_symbol(env.o)

    T = 10000
    for i in tqdm(range(T)):
        play_game(p1, p2, Environment())

    human = Human()
    human.set_symbol(env.x)
    while True:
        # p1.set_verbose(True)
        # p1.set_eps(0)
        # play_game(p1, human, Environment(), draw = 2)
        p2.set_verbose(True)
        p2.set_eps(0)
        play_game(human, p2, Environment(), draw=1)
        answer = input('Play Again? [y/n]:')
        if answer[0] == 'n':
            break


