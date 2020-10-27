import numpy as np
import random
import matplotlib.pyplot as plt

seed =4
random.seed(seed)
np.random.seed(seed)

def generate_random_env(dim=30,mine=100):
    ### -1 means mine, >0 means safe and the # of mines around it
    env = np.zeros((dim,dim))
    random_index = random.sample(range(0,dim*dim),mine)
    for i in random_index:
        env[i//dim,i%dim]=-1
    for i in range(dim):
        for j in range(dim):
            ## count mines in the 8-neighbor
            if env[i,j]!=-1:
                if i-1>=0 and env[i-1,j]==-1:
                    env[i,j]+=1
                if i + 1 < dim and env[i + 1, j] == -1:
                    env[i, j] += 1
                if j-1>=0 and env[i,j-1]==-1:
                    env[i,j]+=1
                if j+1<dim and env[i,j+1]==-1:
                    env[i,j]+=1
                if i-1>=0 and j+1<dim and env[i-1,j+1]==-1:
                    env[i,j]+=1
                if i + 1 < dim and j-1>=0 and env[i + 1, j-1] == -1:
                    env[i, j] += 1
                if j-1>=0 and i-1>=0 and env[i-1,j-1]==-1:
                    env[i,j]+=1
                if j+1<dim and i+1<dim and env[i+1,j+1]==-1:
                    env[i,j]+=1
    return env
class Environment():
    def __init__(self,dim=30,mine=100):
        self.env = generate_random_env(dim,mine)

    def report(self,pos):
        return self.env[pos]

class KnowledgeBase():
    def __init__(self,dim):
        self.dim = dim
        ## in KB, -2 means covered, -1 means mine, >=0 means safe and the # of mines around it, 10 means we only deduce it safe but hidden
        self.state = -2 * np.ones((dim,dim))
        self.deduced_safe_queue_to_click = []
        # # if safe, the number of mines surrounding it indicated by the clue
        # self.clue_mine = -1 * np.ones((dim,dim))
        # number of safe squares identified around it
        self.num_safe_identified = np.zeros((dim,dim))
        # num of mines identified around it
        self.num_mine_identified = np.zeros((dim,dim))
        # num of hidden squares around it
        self.num_hidden = np.zeros((dim,dim))
        for i in range(self.dim):
            for j in range(self.dim):
                if i*(self.dim-1-i) + j*(self.dim-1-j)==0:
                    self.num_hidden[i,j] = 3
                elif i*(self.dim-1-i) * j*(self.dim-1-j)!=0:
                    self.num_hidden[i,j] = 8
                else:
                    self.num_hidden[i,j] = 5
        #### debug
        if 0:
            plt.imshow(self.num_hidden)
            plt.show()
    def inference(self):
        safe_pos = self.all_safe_pos()
        self.deduced_safe_queue_to_click = []
        for pos in safe_pos:
            if self.num_hidden[pos]!=0:
                #### calculate num of neighbors  ( 8 except for edge and corner )
                x,y = pos
                if x*(self.dim-1-x) + y*(self.dim-1-y)==0:
                    sum_safe = 3
                elif x*(self.dim-1-x) * y*(self.dim-1-y) !=0:
                    sum_safe = 8
                else:
                    sum_safe = 5
                #### two constraint rules
                ## mine
                if self.state[pos]-self.num_mine_identified[pos]==self.num_hidden[pos]:
                    self.num_mine_identified[pos]+=self.num_hidden[pos]
                    self.set_hidden_to_mine(pos)
                ## safe
                elif sum_safe-self.state[pos]-self.num_safe_identified[pos]==self.num_hidden[pos]:
                    self.num_safe_identified[pos]+=self.num_hidden[pos]
                    self.deduced_safe(pos)
    def update(self,pos_list,pos_state_list):
        for (pos,pos_state) in zip(pos_list,pos_state_list):
            self.state[pos]=pos_state
            self.set_eight_neighbor_num(pos,pos_state)
    def set_eight_neighbor_num(self,pos,pos_state):
        i,j = pos
        ## neighbor reduce num_hidden by 1 and add num_safe_identified by 1
        self.num_hidden_of_neighbor_reduced_by_1(i,j)
        ## center is mine or safe, reduce neighbor num_safe/mine by 1
        array = self.num_safe_identified if pos_state!=-1 else self.num_mine_identified
        if i - 1 >= 0:
            array[i-1,j]+=1
        if i + 1 < self.dim :
            array[i+1,j]+=1
        if j - 1 >= 0 :
            array[i,j-1]+=1
        if j + 1 < self.dim :
            array[i,j+1]+=1
        if i - 1 >= 0 and j + 1 < self.dim:
            array[i-1,j+1]+=1
        if i + 1 < self.dim and j - 1 >= 0:
            array[i+1,j-1]+=1
        if j - 1 >= 0 and i - 1 >= 0 :
            array[i-1,j-1]+=1
        if j + 1 < self.dim and i + 1 < self.dim :
            array[i+1,j+1]+=1
    def num_hidden_of_neighbor_reduced_by_1(self,i,j):
        ## neighbor reduce num_hidden by 1 and add num_safe_identified by 1
        if i - 1 >= 0 :
            self.num_hidden[i-1,j]-=1
            # self.num_safe_identified[i-1,j]+=1
        if i + 1 < self.dim :
            self.num_hidden[i+1,j]-=1
            # self.num_safe_identified[i + 1, j] += 1
        if j - 1 >= 0 :
            self.num_hidden[i,j-1]-=1
            # self.num_safe_identified[i , j-1] += 1
        if j + 1 < self.dim:
            self.num_hidden[i,j+1]-=1
            # self.num_safe_identified[i, j+1] += 1
        if i - 1 >= 0 and j + 1 < self.dim :
            self.num_hidden[i-1,j+1]-=1
            # self.num_safe_identified[i - 1, j+1] += 1
        if i + 1 < self.dim and j - 1 >= 0 :
            self.num_hidden[i+1,j-1]-=1
            # self.num_safe_identified[i + 1, j-1] += 1
        if j - 1 >= 0 and i - 1 >= 0 :
            self.num_hidden[i-1,j-1]-=1
            # self.num_safe_identified[i - 1 , j-1] += 1
        if j + 1 < self.dim and i + 1 < self.dim :
            self.num_hidden[i+1,j+1]-=1
            # self.num_safe_identified[i + 1, j+1] += 1

    def set_hidden_to_mine(self,pos):
        # value
        i,j = pos
        print("mark as mine: ")
        if i - 1 >= 0 and self.state[i - 1, j] == -2:
            self.state[i - 1, j] = -1
            self.num_hidden_of_neighbor_reduced_by_1(i-1,j)
            print((i-1,j))
        if i + 1 < self.dim and self.state[i + 1, j] == -2:
            self.state[i + 1, j] = -1
            self.num_hidden_of_neighbor_reduced_by_1(i+1,j)
            print((i + 1, j))
        if j - 1 >= 0 and self.state[i, j - 1] == -2:
            self.state[i, j - 1] = -1
            self.num_hidden_of_neighbor_reduced_by_1(i,j-1)
            print((i , j- 1))
        if j + 1 < self.dim and self.state[i, j + 1] == -2:
            self.state[i, j + 1] = -1
            self.num_hidden_of_neighbor_reduced_by_1(i,j+1)
            print((i, j + 1))
        if i - 1 >= 0 and j + 1 < self.dim and self.state[i - 1, j + 1] == -2:
            self.state[i - 1, j + 1] = -1
            self.num_hidden_of_neighbor_reduced_by_1(i-1,j+1)
            print((i-1, j + 1))
        if i + 1 < self.dim and j - 1 >= 0 and self.state[i + 1, j - 1] == -2:
            self.state[i + 1, j - 1] = -1
            self.num_hidden_of_neighbor_reduced_by_1(i+1,j-1)
            print((i + 1, j - 1))
        if j - 1 >= 0 and i - 1 > 0 and self.state[i - 1, j - 1] == -2:
            self.state[i - 1, j - 1] = -1
            self.num_hidden_of_neighbor_reduced_by_1(i-1,j-1)
            print((i - 1, j - 1))
        if j + 1 < self.dim and i + 1 < self.dim and self.state[i + 1, j + 1] == -2:
            self.state[i + 1, j + 1] = -1
            self.num_hidden_of_neighbor_reduced_by_1(i+1,j+1)
            print((i + 1, j + 1))
    def deduced_safe(self,pos):
        i,j = pos
        if i - 1 >= 0 and self.state[i - 1, j] == -2:
            self.deduced_safe_queue_to_click.append((i-1,j))
        if i + 1 < self.dim and self.state[i + 1, j] == -2:
            self.deduced_safe_queue_to_click.append((i+1,j))
        if j - 1 >= 0 and self.state[i, j - 1] == -2:
            self.deduced_safe_queue_to_click.append((i,j-1))
        if j + 1 < self.dim and self.state[i, j + 1] == -2:
            self.deduced_safe_queue_to_click.append((i,j+1))
        if i - 1 >= 0 and j + 1 < self.dim and self.state[i - 1, j + 1] == -2:
            self.deduced_safe_queue_to_click.append((i-1,j+1))
        if i + 1 < self.dim and j - 1 >= 0 and self.state[i + 1, j - 1] == -2:
            self.deduced_safe_queue_to_click.append((i+1,j-1))
        if j - 1 >= 0 and i - 1 > 0 and self.state[i - 1, j - 1] == -2:
            self.deduced_safe_queue_to_click.append((i-1,j-1))
        if j + 1 < self.dim and i + 1 < self.dim and self.state[i + 1, j + 1] == -2:
            self.deduced_safe_queue_to_click.append((i+1,j+1))
    def all_safe_pos(self):
        safe_x, safe_y = np.where(self.state >=0)
        return [(i, j) for (i, j) in zip(safe_x, safe_y)]
    def all_hidden_pos(self):
        hidden_x,hidden_y = np.where(self.state==-2)
        return [(i,j) for (i,j) in zip(hidden_x,hidden_y)]
    def get_next_round_clicks(self):
        self.inference()
        if len(self.deduced_safe_queue_to_click)!=0:
            return self.deduced_safe_queue_to_click
        else:
            all_hidden = self.all_hidden_pos()
            if len(all_hidden)==0:
                return []
            else:
                return random.sample(self.all_hidden_pos(),1)

class Agent():
    def __init__(self,dim,n):
        self.KB = KnowledgeBase(dim)
        self.env = Environment(dim,n)
    def explore(self,pos_list):
        return [self.env.report(pos) for pos in pos_list]

    def strategy_BL(self):
        print("start")
        round = 0
        explore_pos = self.KB.get_next_round_clicks()

        while(len(explore_pos)!=0):
            round+=1
            print("round: ",round,"explore :",explore_pos)

            explore_pos_state = self.explore(explore_pos)
            self.KB.update(explore_pos,explore_pos_state)
            explore_pos = self.KB.get_next_round_clicks()
            #### debug
            if 1:
                plt.subplot(141)
                plt.imshow(self.env.env)
                plt.subplot(142)
                plt.imshow(self.KB.state)
                plt.subplot(143)
                plt.imshow(self.KB.num_hidden)
                plt.subplot(144)
                plt.imshow(self.KB.num_safe_identified)
                plt.show()
        print("finsh")



