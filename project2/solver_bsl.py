import numpy as np
import random
import matplotlib.pyplot as plt

def get_Neighbors( cell,dim):
    x, y = cell
    neighbors = set()
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            else:
                if 0 <= x + i < dim and 0 <= y + j < dim:
                    neighbors.add((x + i, y + j))
    return neighbors

class Environment():
    def __init__(self,dim=30,mine=100):
        self.board = self.generate_random_env(dim,mine)

    def generate_random_env(self,dim=30, mine=100):
        ### -1 means mine, >0 means safe and the # of mines around it
        env = np.zeros((dim, dim),np.int)
        random_index = random.sample(range(0, dim * dim), mine)
        for i in random_index:
            env[i // dim, i % dim] = -1
        for i in range(dim):
            for j in range(dim):
                ## count mines in the 8-neighbor
                if env[i, j] != -1:
                    neighbors = get_Neighbors((i,j),dim)
                    for n in neighbors:
                        if env[n]==-1:
                            env[i,j]+=1
        return env

    def report(self,pos):
        return self.board[pos]
class Sentence():
    def __init__(self,num_neighbor, num_safes, num_mines, hidden,clue):
        self.num_mines = num_mines
        self.num_safes = num_safes
        self.hidden = hidden
        self.count = clue
        self.num_neighbors = num_neighbor

    def conclude_mines(self):
        # determine hidden cells are mines
        if self.count - self.num_mines == len(self.hidden):
            return self.hidden
        else:
            return set()
    def conclude_safes(self):
        # determine hidden cells are safe
        if self.num_neighbors - self.count - self.num_safes == len(self.hidden):
            return self.hidden
        else:
            return set()

    def mark_mine(self,cell):
        # using new information to update a sentence (mines)
        if cell in self.hidden:
            self.hidden.remove(cell)
            self.num_mines+=1

    def mark_safe(self,cell):
        # using new information to update a sentence (safe)
        if cell in self.hidden:
            self.hidden.remove(cell)
            self.num_safes+=1

class Agent():
    def __init__(self,dim):
        self.dim = dim
        self.cells_visited = set()
        # all cells known to be mines
        self.mines = set()
        # all cells known to be safe
        self.safes = set()
        # all known sentence
        self.knowledge = []
        # info get from env
        self.cells_info_from_env = dict()
    def mark_mine(self,cell):
        ## add a cell to self.mines and inform all sentense in self.knowledge
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)
    def mark_safe(self,cell):
        ## add a cell to self.safes and inform all sentense in self.knowledge
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)
    def strategy(self,cell,count):
        ## improved strategy
        ## after reveal a safe cell, add the cell to knowledge base, update knowledge base and then inference new sentence
        self.cells_visited.add(cell)
        self.cells_info_from_env[cell] = count
        if count==-1:
            self.mark_mine(cell)
        else:
            self.mark_safe(cell)
            neighbor_cells = get_Neighbors(cell,self.dim)
            hidden = set()
            num_safes = 0
            num_mines = 0
            for n in neighbor_cells:
                if n in self.safes:
                    num_safes+=1
                elif n in self.mines:
                    num_mines+=1
                else:
                    hidden.add(n)
            self.knowledge.append(Sentence(len(neighbor_cells), num_safes,num_mines,hidden,count))
        self.update_knowledge()
        # new_knowledge = self.inference()
        # while len(new_knowledge):
        #     # print('current knowledge: ',len(self.knowledge))
        #     # for mm in self.knowledge:
        #     #     print(mm.cell_set,mm.count)
        #     # # print('new knowledge: ', len(new_knowledge))
        #     # for mm in new_knowledge:
        #     #     print(mm.cell_set,mm.count)
        #     for n in new_knowledge:
        #         self.knowledge.append(n)
        #
        #     self.update_knowledge()
        #
        #     new_knowledge = self.inference()

    def update_knowledge(self):

        flag_revisit = 1
        # we need to revisit whole sentence again if any sentence has been updated in the loop
        while flag_revisit:
            safe = set()
            mine = set()
            flag_revisit = 0
            for sentence in self.knowledge:
                for cell in sentence.conclude_safes():
                    safe.add(cell)
                    # self.mark_safe(cell)
                    flag_revisit = 1
                for cell in sentence.conclude_mines():
                    mine.add(cell)
                    # self.mark_mine(cell)
                    flag_revisit = 1
            for s in safe:
                self.mark_safe(s)
            for m in mine:
                self.mark_mine(m)
            ## remove empty sentence in knowledge base
            empty_sentence_list = []
            for s in self.knowledge:
                if len(s.hidden)==0:
                    empty_sentence_list.append(s)
            self.knowledge = [ s for s in self.knowledge if s not in empty_sentence_list ]


    # def inference(self):
    #     new_knowledge = set()
    #     for s1 in self.knowledge:
    #         for s2 in self.knowledge:
    #             if s1.cell_set == s2.cell_set\
    #                     and s1.count == s2.count:
    #                 continue
    #             if s1.cell_set.issubset(s2.cell_set):
    #
    #                 new_s = Sentence(s2.cell_set-s1.cell_set,s2.count-s1.count)
    #                 if new_s not in self.knowledge:
    #                     new_knowledge.add(new_s)
    #     return new_knowledge

    def explore(self):
        # next cell to click, safe move first, if no safe infered, random move
        safe_not_visited = self.safes - self.cells_visited
        if len(safe_not_visited):
            return safe_not_visited.pop()
        else:
            random_pool = []
            for i in range(self.dim):
                for j in range(self.dim):
                    if (i,j) not in (self.cells_visited.union(self.mines)):
                        random_pool.append((i,j))
            if len(random_pool) == 0:
                return None
            else:
                return random.sample(random_pool,1)[0]
    def score(self,num_mines):
        assert  num_mines!=0
        ## call it only when you know all mines has been either marked or revealed
        return len(self.mines - self.cells_visited) /  num_mines

    def __repr__(self):
        return "baseline"




