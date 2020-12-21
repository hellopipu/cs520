import numpy as np
import random
import matplotlib.pyplot as plt
from solver_bsl import get_Neighbors
from solver_bsl import Agent as Agent_bsl



class Sentence():
    def __init__(self,cells,count):
        self.cell_set=set(cells)
        # how many cells in the set are mines
        self.count = count
    def __eq__(self, other):
        return self.cell_set == other.cell_set and self.count == other.count
    def __hash__(self):
        return hash((tuple(self.cell_set),self.count))
    def conclude_mines(self):
        # determine any of the cells in the set are mines
        if len(self.cell_set)==self.count:
            return self.cell_set
        else:
            return set()
    def conclude_safes(self):
        # determine any of the cells in the set are safe
        if self.count==0:
            return self.cell_set
        else:
            return set()

    def mark_mine(self,cell):
        # using new information to update a sentence (mines)
        if cell in self.cell_set:
            self.cell_set.remove(cell)
            self.count-=1
    def mark_safe(self,cell):
        # using new information to update a sentence (safe)
        if cell in self.cell_set:
            self.cell_set.remove(cell)

class Agent(Agent_bsl):
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
            neighbor_known = set()
            for c in neighbor_cells:
                if c in self.safes:
                    neighbor_known.add(c)
                elif c in self.mines:
                    count-=1
                    neighbor_known.add(c)
            neighbor_cells-=neighbor_known
            self.knowledge.append(Sentence(neighbor_cells, count))
        self.update_knowledge()
        new_knowledge = self.inference()
        while len(new_knowledge):
            # print('current knowledge: ',len(self.knowledge))
            # for mm in self.knowledge:
            #     print(mm.cell_set,mm.count)
            # # print('new knowledge: ', len(new_knowledge))
            # for mm in new_knowledge:
            #     print(mm.cell_set,mm.count)
            for n in new_knowledge:
                self.knowledge.append(n)

            self.update_knowledge()

            new_knowledge = self.inference()

    def update_knowledge(self):

        flag_revisit = 1
        # we need to revisit whole sentence again if any sentence has been updated in the loop
        while flag_revisit:
            flag_revisit = 0
            safe = set()
            mine = set()
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
                if len(s.cell_set)==0:
                    empty_sentence_list.append(s)
            self.knowledge = [ s for s in self.knowledge if s not in empty_sentence_list ]


    def inference(self):
        new_knowledge = set()
        for s1 in self.knowledge:
            for s2 in self.knowledge:
                if s1.cell_set == s2.cell_set\
                        and s1.count == s2.count:
                    continue
                if s1.cell_set.issubset(s2.cell_set):

                    new_s = Sentence(s2.cell_set-s1.cell_set,s2.count-s1.count)
                    if new_s not in self.knowledge:
                        new_knowledge.add(new_s)
        return new_knowledge

    def __repr__(self):
        return "improve"






