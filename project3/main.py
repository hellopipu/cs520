import random
import numpy as np
import click
from progress.bar import Bar
import copy
def generate_random_map(dim):
    terrain_prob = {"flat":0.2, "hilly":0.3, "forest":0.3, "caves": 0.2}
    map = np.zeros((dim,dim),dtype=np.int)
    for i in range(dim):
        for j in range(dim):
            rand_prob = np.random.rand()
            if rand_prob <=terrain_prob["flat"]:
                map[i][j] = 0
            elif rand_prob <= terrain_prob["flat"] + terrain_prob["hilly"]:
                map[i][j] = 1
            elif rand_prob <= terrain_prob["flat"] + terrain_prob["hilly"] + terrain_prob["forest"]:
                map[i][j] = 2
            else:
                map[i][j] = 3

    return map

class Map():
    def __init__(self,dim):
        self.dim = dim
        self.map = generate_random_map(dim)
        self.terrain = ["flat", "hilly", "forest", "caves"]
        self.FN_rates = [0.1, 0.3, 0.7, 0.9]
        self.target_i = np.random.randint(dim)
        self.target_j = np.random.randint(dim)

    def output_2_search(self,i,j):
        istarget = True if i==self.target_i and j==self.target_j else False
        if not istarget:
            return False
        else:
            if np.random.rand() < self.FN_rates[self.map[i,j]]:
                return False
            else:
                return True
    def reset_target(self):
        self.target_i = np.random.randint(self.dim)
        self.target_j = np.random.randint(self.dim)





class BasicAgent():
    def __init__(self,dim):
        self.dim = dim
        self.terrain = ["flat","hilly","forest","caves"]
        self.FP_rates = [0.1, 0.3, 0.7, 0.9]
        self.belif_array = np.ones((dim,dim),dtype=np.float32) /dim /dim
        self.confid_array = np.zeros((dim,dim))
        self.map = Map(dim)
        for i in range(dim):
            for j in range(dim):
                self.confid_array[i,j] = self.belif_array[i,j] * (1-self.FP_rates[self.map.map[i,j]] )
    def find_target(self,rule):
        receive_signal = False
        cnt = 0
        while not receive_signal:
            cnt += 1
            receive_signal, i,j = self.determine_cell_and_search(rule)
            if receive_signal:
                return cnt
            else:
                self.update(i,j)

    def determine_cell_and_search(self,rule):
        if rule == 'belief':
            i,j = np.where(self.belif_array==np.max(self.belif_array))
        else:
            i, j = np.where(self.confid_array == np.max(self.confid_array))
        index = np.random.randint(0,len(i))
        i,j = i[index],j[index]
        return self.map.output_2_search(i,j),i,j

    def update(self,i,j):
        self.belif_array[i,j] = self.belif_array[i,j] * self.FP_rates[self.map.map[i,j]]
        self.belif_array = self.belif_array/self.belif_array.sum()
        for i in range(self.dim):
            for j in range(self.dim):
                self.confid_array[i, j] = self.belif_array[i, j] * (1 - self.FP_rates[self.map.map[i, j]])

class Agent(BasicAgent):
    def __init__(self,dim):
        super().__init__(dim)
        self.manhanttan = np.zeros((self.dim,self.dim))
    def find_target(self,rule):
        receive_signal = False
        cnt = 1
        point_i = np.random.randint(0, self.dim)
        point_j = np.random.randint(0, self.dim)

        while not receive_signal:
            receive_signal, point_i,point_j, min_add = self.determine_cell_and_search(rule,point_i,point_j)
            if receive_signal:
                return cnt
            else:
                cnt += min_add
                self.update(point_i,point_j)
    def find_nearest(self,i,j,point_i,point_j,rule):
        min_temp =  float("inf")

        if rule=='belief' or rule == 'confid':
            for ii,jj in zip(i,j):
                if abs(ii-point_j) + abs(jj-point_j) < min_temp:
                    min_temp = abs(ii-point_i) + abs(jj-point_j)
                    point = [ii,jj]
        elif rule == 'manhattan':
            for m in range(self.dim):
                for n in range(self.dim):
                    self.manhanttan[m,n] = 1.* (abs(m-point_i) + abs(n-point_j)+1) / (self.confid_array[m,n]+np.finfo(np.float32).eps)
                    # if self.manhanttan[m,n]==0:
                    #     self.manhanttan[m, n] = float("inf")
            ii, jj = np.where(self.manhanttan == np.min(self.manhanttan) )
            index = np.random.randint(0,len(ii))
            ii,jj = ii[index],jj[index]
            min_temp = abs(ii-point_i) + abs(jj-point_j)
            point = [ii, jj]
        else:
            for m in range(self.dim):
                for n in range(self.dim):
                    self.manhanttan[m, n] = 1. * 0.99**(abs(m - point_i) + abs(n - point_j) ) * (
                                self.confid_array[m, n] )
            ii, jj = np.where(self.manhanttan == np.max(self.manhanttan))
            index = np.random.randint(0, len(ii))
            ii, jj = ii[index], jj[index]

            # belif_array = copy.deepcopy(self.belif_array)
            # confid_array = np.zeros((self.dim,self.dim))
            # belif_array[ii, jj] = self.belif_array[ii, jj] * self.FP_rates[self.map.map[ii, jj]]
            # belif_array = belif_array / belif_array.sum()
            # for i in range(self.dim):
            #     for j in range(self.dim):
            #         confid_array[i, j] = belif_array[i, j] * (1 - self.FP_rates[self.map.map[i, j]])
            #
            # for m in range(self.dim):
            #     for n in range(self.dim):
            #         self.manhanttan[m, n] = 1. * (abs(m - point_i) + abs(n - point_j)+1 ) / (
            #                     confid_array[m, n] + np.finfo(np.float32).eps)
            # ii, jj = np.where(self.manhanttan == np.min(self.manhanttan))
            # index = np.random.randint(0, len(ii))
            # ii, jj = ii[index], jj[index]
            min_temp = abs(ii - point_i) + abs(jj - point_j)
            point = [ii, jj]

        return point[0],point[1], min_temp+1



    def determine_cell_and_search(self,rule,point_i,point_j):
        if rule == 'belief':
            i,j = np.where(self.belif_array==np.max(self.belif_array))
        else:
            i, j = np.where(self.confid_array == np.max(self.confid_array))

        i,j,min_add = self.find_nearest(i,j,point_i,point_j,rule)
        return self.map.output_2_search(i,j),i,j,min_add

    def update(self,i,j):
        self.belif_array[i,j] = self.belif_array[i,j] * self.FP_rates[self.map.map[i,j]]
        self.belif_array = self.belif_array/self.belif_array.sum()
        for i in range(self.dim):
            for j in range(self.dim):
                self.confid_array[i, j] = self.belif_array[i, j] * (1 - self.FP_rates[self.map.map[i, j]])


@click.command()
@click.option('--dim',default = 5,type = int)
@click.option('--mode',default = 'point3', type=click.Choice([ 'point3' , 'point4' ]))
@click.option('--repeat',default = 50,type = int)
def main(dim,repeat,mode):

    if mode == 'point3':
        '''
        experiment for point 3
        '''
        a = BasicAgent(dim)
        belief_cnt = 0
        confid_cnt = 0
        with Bar('Processing',max = repeat) as bar:
            for i in range(repeat):
                belief_cnt += a.find_target('belief')
                confid_cnt += a.find_target('confid')
                a.map.reset_target()
                bar.next()
        belief_cnt /= repeat
        confid_cnt /= repeat

        print('belief : ',belief_cnt)
        print('confid : ',confid_cnt)
    else:
        '''
        experiment for point 4
        '''
        a = Agent(dim)
        agent1_cnt = 0
        agent2_cnt = 0
        agent3_cnt = 0
        agent_imp_cnt = 0
        with Bar('Processing', max=repeat) as bar:
            for i in range(repeat):
                agent1_cnt += a.find_target('belief')
                agent2_cnt += a.find_target('confid')
                agent3_cnt += a.find_target('manhattan')
                agent_imp_cnt += a.find_target('manhattan2')
                a.map.reset_target()
                bar.next()
        agent1_cnt /= repeat
        agent2_cnt /= repeat
        agent3_cnt /= repeat
        agent_imp_cnt /= repeat

        print('agent1 : ', agent1_cnt)
        print('agent2 : ', agent2_cnt)
        print('agent3 : ', agent3_cnt)
        print('agent_imp : ', agent_imp_cnt)


main()
# dim=10
# belief :  498.62
# confid :  126.14

# dim=20
# belief :  1701.86
# confid :  625.52


#####
# agent1 :  2662.68
# agent2 :  1129.46


