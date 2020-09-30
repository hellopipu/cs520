import numpy as np
import matplotlib as mpl
# mpl.use("Agg")
import matplotlib.pyplot as plt
import  time
import copy
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
class timer(object):
    '''
    wrapper, calculate time of the function.
    '''
    def __init__(self, verbose=1):
        self.verbose = verbose

    def __call__(self, fn):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = fn(*args, **kwargs)
            if self.verbose:
                print('Function "%s" costs %.2f s' %(fn.__name__, time.time() - start))
            return result
        return wrapper

class Stack(object):
    '''
    stack structure
    '''
    def __init__(self):
        self.stack = []
    def push(self,data):
        self.stack.append(data)
    def pop(self):
        return self.stack.pop()
    def gettop(self):
        return self.stack[-1]
    def __len__(self):
        return len(self.stack)
class Queue(object):
    '''
    queque structure
    '''
    def __init__(self):
        self.queque = []
    def add(self,data):
        self.queque.append(data)
    def pop(self):
        return self.queque.pop(0)
    def getpeek(self):
        return self.queque[0]
    def __len__(self):
        return len(self.queque)
class Prior_Queue(Queue):
    '''
    queque structure
    '''
    def add(self,data):
        '''
        add in order of 'prior'
        :param data:
        :return:
        '''
        length = len(self.queque)
        if length==0:
            self.queque.append(data)
        else:
            for i in range(length):
                if data['prior']<self.queque[i]['prior']:
                    self.queque.insert(i,data)
                    break
                else:
                    if i==length-1:
                        self.queque.append(data)
                    else:
                        continue

def generate_random_matrix(dim,p,cnt_initial_fire=10):
    '''
    generate random matrix (may be not solvable), value = 2 is empty, value = 1 is on fire, value = 0 is occupied
    :param dim: dimension of the maze matrix
    :param p:   probability of a cell being occupied
    :return: random matrix
    '''
    assert (type(dim)==int and dim>1)
    assert (p>=0 and p<=1.)
    while 1:
        maze_matrix = None
        fire_point_list = []
        #### generate walls
        while not DFS_search(maze_matrix, [dim - 1, dim - 1]):
            #### use DFS to validate the generated maze
            maze_matrix = 2*np.ones((dim, dim), dtype=np.uint8)
            for rows in range(dim):
                for cols in range(dim):
                    if np.random.rand() <= p:
                        maze_matrix[rows, cols] = 0
            maze_matrix[0, 0] = 2
            maze_matrix[dim - 1, dim - 1] = 2
        #### generate one fire point
        while(len(fire_point_list)!=cnt_initial_fire):
            cnt_tolerance = 0
            tolerance_flag = 0
            maze_matrix_fire = None
            fire_point = None
            while not DFS_search(maze_matrix_fire, fire_point, ignore_fire=True) or not DFS_search(maze_matrix_fire, [dim - 1, dim - 1]):
                #### use DFS to validate the generated maze with fire
                maze_matrix_fire = copy.deepcopy(maze_matrix)
                while fire_point is None or fire_point == [0, 0] or fire_point == [dim - 1, dim - 1] or maze_matrix[fire_point[0],fire_point[1]]!=2:
                    fire_point = np.random.choice(dim, 2, replace=True)
                    fire_point = [fire_point[0], fire_point[1]]
                maze_matrix_fire[fire_point[0], fire_point[1]] = 1
                cnt_tolerance += 1
                if cnt_tolerance > 200:
                    tolerance_flag = 1
                    break
            if tolerance_flag == 1:
                break
            else:
                if fire_point not in fire_point_list:
                    fire_point_list.append(fire_point)

        if len(fire_point_list)==cnt_initial_fire:
            break
    return maze_matrix,fire_point_list

def get_next_choice(matrix,current_pos,history, ignore_fire = False):
    '''

    :param matrix: maze matrix
    :param current_pos: current position
    :param history: positions have been visited
    :param ignore_fire: ignore fire source, for searching path to the fire source
    :return: list, next step valiable moves
    '''
    next_choice = []
    dim = matrix.shape[0]
    a,b = current_pos[0],current_pos[1]
    flag = 0 if ignore_fire else 1
    if ([a-1,b] not in history) and a-1>=0 and matrix[a-1,b] > flag:
        next_choice.append([a-1,b])

    if ([a+1,b] not in history) and a+1<dim and matrix[a+1,b] > flag:
        next_choice.append([a+1,b])

    if ([a,b+1] not in history) and b+1<dim and matrix[a,b+1] > flag:
        next_choice.append([a,b+1])

    if ([a,b-1] not in history) and b-1>=0 and matrix[a,b-1] > flag:
        next_choice.append([a,b-1])

    return next_choice

def DFS_search(matrix,goal_point,ignore_fire = False):
    '''
    Depth First Search from [0,0] to goal point
    :param matrix: matrix for searching
    :param goal_point: lower right or fire source
    :param ignore_fire: ignore fire source
    :return: solvable or not
    '''
    if matrix is None or goal_point is None:
        return []
    start_point = [0,0]
    stack = Stack()
    stack.push(start_point)
    history = []
    flag_deep_path = 0
    parents_dict = {}
    while len(stack):
        current_pos = stack.pop()
        # print('current_pos',current_pos)
        if current_pos in history:
            continue
        else:
            history.append(current_pos)
            # print('append_pos', current_pos)
        next_choice = get_next_choice(matrix, current_pos, history, ignore_fire)


        for i in next_choice:
            parents_dict[str(i)] = current_pos
            if i == goal_point:
                #### find the goal, break the loop
                flag_deep_path = 1
                break
            stack.push(i)
        if flag_deep_path == 1:
            break
    if flag_deep_path == 1:
        ### refind the path
        last_node = goal_point
        deep_path = [last_node]
        # print(parents_dict)
        while 1:
            deep_path.append(parents_dict[str(last_node)])
            last_node = parents_dict[str(last_node)]
            if deep_path[-1] == start_point:
                break
        return deep_path[::-1]
    else:
        return []


def search_valid(maze_matrix):
    '''
    make sure maze is solvable and there's path from agent to the initial fire source
    :param maze_matrix: maze matrix
    :return:
    '''
    if maze_matrix is None:
        return False
    ## if valid for goal point && fire point
    dim = maze_matrix.shape[0]
    fire_point_array = np.argwhere(maze_matrix==1)

    return DFS_search(maze_matrix,[dim-1,dim-1]) and DFS_search(maze_matrix,list(fire_point_array[0]),ignore_fire=True)
## initiate maze
@timer(False)
def generate_maze(dim,p,cnt_initial_fire = 10):
    '''
    generate valid maze matrix, value = 2 is empty, value = 1 is on fire, value = 0 is occupied
    :param dim: dimension of the maze matrix
    :param p:   probability of a cell being occupied
    :return:    maze matrix
    '''
    maze_matrix = None
    # while not search_valid(maze_matrix):
    maze_matrix,fire_source_list = generate_random_matrix(dim,p,cnt_initial_fire)
    return maze_matrix,fire_source_list

def generate_advanced_maze(maze_matrix_,q,step = 0):
    '''
    advanced maze at a specific step
    :param maze_matrix:
    :param q:
    :param step:
    :return: list of new fire list of each step
    '''
    list_new_fire= []
    maze_matrix =copy.deepcopy(maze_matrix_)
    fire_list = set([tuple(p) for p in np.argwhere(maze_matrix == 1)])
    neighbor_dict = {}
    already_fire = copy.deepcopy(fire_list)
    list_current_new_fire = fire_list
    for _ in range(step):
        dim = maze_matrix.shape[0]
        for fire_point in list_current_new_fire:
            for nn in get_neighbor(fire_point, dim):
                if maze_matrix[nn]==2 and nn not in already_fire:
                    neighbor_dict[nn] = neighbor_dict.get(nn,0)+1
        list_current_new_fire = set()
        for nn in neighbor_dict:
            if np.random.rand() <= 1 - (1 - q) ** neighbor_dict[nn]:
                list_current_new_fire.add(nn)
        for kk in list_current_new_fire:
            neighbor_dict.pop(kk)
        list_new_fire.append(list_current_new_fire)
        already_fire = already_fire.union(list_current_new_fire)
    # print('generate',list_new_fire)
    return list_new_fire


@timer(False)
def BFS_search(matrix,start_point):
    '''
    Breadth First Search to find the shortest path in the maze
    :param matrix: maze matrix
    :param start_point:
    :return: path list, [start point, next step, ..., gaol point]
    '''
    # print('matrix: ',matrix)

    if matrix is None :
        return False
    dim = matrix.shape[0]
    goal_point = [dim-1,dim-1]
    queue = Queue()
    queue.add(start_point)
    history = []
    flag_shortest_path = 0
    parents_dict={}
    while len(queue):
        current_pos = queue.pop()
        if current_pos in history:
            continue
        else:
            history.append(current_pos)

        next_choice = get_next_choice(matrix, current_pos, history)
        for i in next_choice:
            parents_dict[str(i)] = current_pos
            if i== goal_point:
                #### find the goal, break the loop
                flag_shortest_path = 1
                break
            queue.add(i)
        if flag_shortest_path ==1:
            break
    if flag_shortest_path == 1:
        ### refind the path
        last_node = goal_point
        shortest_path = [last_node]
        # print(parents_dict)
        while 1:
            shortest_path.append(parents_dict[str(last_node)])
            last_node = parents_dict[str(last_node)]
            if shortest_path[-1]==start_point:
                break
        return shortest_path[::-1]
    else:
        return []
def get_neighbor(pos,dim):
    neighbor_list = []
    if pos[0]-1>=0:
        neighbor_list.append((pos[0]-1,pos[1]))
    if pos[0]+1<dim:
        neighbor_list.append((pos[0]+1,pos[1]))
    if pos[1]-1>=0:
        neighbor_list.append((pos[0],pos[1]-1))
    if pos[1]+1<dim:
        neighbor_list.append((pos[0],pos[1]+1))
    return neighbor_list

def generate_cost_map(matrix_,fire_point_list,q):
    matrix = copy.deepcopy(matrix_)
    dim = matrix.shape[0]
    matrix[matrix!=1]=0
    cost_map = copy.deepcopy(matrix)
    # for i in range(1,4):
    #     for j in range(1,dim-1):
    #         for  k in range(1,dim-1):
    #             if cost_map[j,k]==i:
    #                 cost_map[j-1, k] = i+1 if cost_map[j-1,k]==0 else cost_map[j-1,k]
    #                 cost_map[j, k-1] = i + 1 if cost_map[j, k-1] == 0 else cost_map[j, k-1]
    #                 cost_map[j+1, k] = i + 1 if cost_map[j+1, k] == 0 else cost_map[j+1, k]
    #                 cost_map[j, k+1] = i + 1 if cost_map[j, k+1] == 0 else cost_map[j, k+1]
    while 0 in cost_map:
        l_new_fire = []
        for i in fire_point_list:
            for j in get_neighbor(i,dim):
                if cost_map[j] == 0:
                    l_new_fire.append(j)
                    cost_map[j] = cost_map[j] + 1
        fire_point_list = copy.deepcopy(l_new_fire)
    cost_map[cost_map==1]=2
    cost_map = np.exp(10*q**cost_map)
    return cost_map



def Astar_search_future(matrix, start_point,fire_point_list,q):
    # generate_cost_map(matrix)
    '''
    Astar Search
    :param matrix:
    :param start_point:
    :return:
    '''
    if matrix is None :
        return False
    dim = matrix.shape[0]
    goal_point = [dim-1,dim-1]
    cost_map = generate_cost_map(matrix,fire_point_list,q)
    queue = Prior_Queue()
    queue.add({'pos':start_point,'dist':cost_map[start_point[0],start_point[1]],'heur':2*(dim-1)-start_point[0]-start_point[1], 'prior':2*(dim-1)-start_point[0]-start_point[1]+cost_map[start_point[0],start_point[1]]})
    history = []
    flag_shortest_path = 0
    parents_dict={}
    while len(queue):
        current_node = queue.pop()
        current_pos = current_node['pos']
        current_dist = current_node['dist']
        if current_pos in history:
            continue
        else:
            history.append(current_pos)
            # print(current_node['prior'])

        next_choice = get_next_choice(matrix, current_pos, history)
        for i in next_choice:
            parents_dict[str(i)] = current_pos
            if i== goal_point:
                #### find the goal, break the loop
                flag_shortest_path = 1
                break
            queue.add({'pos':i,'dist':current_dist+cost_map[i[0],i[1]],'heur':2*(dim-1)-i[0]-i[1], 'prior':(2*(dim-1)-i[0]-i[1]+ current_dist+cost_map[i[0],i[1]])})
        if flag_shortest_path ==1:
            break
    if flag_shortest_path == 1:
        ### refind the path
        last_node = goal_point
        shortest_path = [last_node]
        # print(parents_dict)
        while 1:
            shortest_path.append(parents_dict[str(last_node)])
            last_node = parents_dict[str(last_node)]
            if shortest_path[-1]==start_point:
                break
        return shortest_path[::-1]
    else:
        return []

@timer(False)
def Astar_search(matrix, start_point):
    # generate_cost_map(matrix)
    '''
    Astar Search
    :param matrix:
    :param start_point:
    :return:
    '''
    if matrix is None :
        return False
    dim = matrix.shape[0]
    goal_point = [dim-1,dim-1]
    queue = Prior_Queue()
    queue.add({'pos':start_point,'dist':0,'heur':2*(dim-1)-start_point[0]-start_point[1], 'prior':2*(dim-1)-start_point[0]-start_point[1]})
    history = []
    flag_shortest_path = 0
    parents_dict={}
    while len(queue):
        current_node = queue.pop()
        current_pos = current_node['pos']
        current_dist = current_node['dist']
        if current_pos in history:
            continue
        else:
            history.append(current_pos)
            # print(current_node['prior'])

        next_choice = get_next_choice(matrix, current_pos, history)
        for i in next_choice:
            parents_dict[str(i)] = current_pos
            if i== goal_point:
                #### find the goal, break the loop
                flag_shortest_path = 1
                break
            queue.add({'pos':i,'dist':current_dist+1,'heur':2*(dim-1)-i[0]-i[1], 'prior':(2*(dim-1)-i[0]-i[1]+ current_dist+1)})
        if flag_shortest_path ==1:
            break
    if flag_shortest_path == 1:
        ### refind the path
        last_node = goal_point
        shortest_path = [last_node]
        # print(parents_dict)
        while 1:
            shortest_path.append(parents_dict[str(last_node)])
            last_node = parents_dict[str(last_node)]
            if shortest_path[-1]==start_point:
                break
        return shortest_path[::-1]
    else:
        return []

# @timer(False)
def strategy_one(maze_init_,q,advanced_maze):
    '''
    find the shortest path (BFS) of the initial maze, and ignore the fire advance.
    :param maze_init:
    :param q:
    :return:
    '''
    maze_init = copy.deepcopy(maze_init_)
    shortest_path = Astar_search(maze_init,[0,0])
    # advanced_maze = generate_advanced_maze(maze_init,q,step = len(shortest_path)-1)
    for maze_new_fire,pos in zip(advanced_maze,shortest_path[1::]):
        for fire in maze_new_fire:
            maze_init[fire]=1
        if maze_init[pos[0],pos[1]] == 2:
            continue
        elif maze_init[pos[0],pos[1]] == 0:
            print("warning...its occupied space, you should not be there")
        elif maze_init[pos[0], pos[1]] == 1:
            return False
    return True

# @timer(False)
def strategy_two(maze_init_,q, advanced_maze):
    '''
    recompute the shortest path (BFS) of the current maze at each step.
    :param maze_init:
    :param q:
    :return:
    '''
    maze_init = copy.deepcopy(maze_init_)
    start_point = [0, 0]
    dim = maze_init.shape[0]
    current_step = 0
    while 1:
        shortest_path = Astar_search(maze_init, start_point)
        if len(shortest_path)==0:
            return False
        next_point = shortest_path[1]
        # advanced_maze = generate_advanced_maze(maze_init,q,step = 1)[0]
        for new_fire in advanced_maze[current_step]:
            maze_init[new_fire]=1
        if maze_init[next_point[0],next_point[1]] == 1:
            return False
        if next_point == [dim-1,dim-1]:
            return True
        start_point = next_point
        current_step += 1

def strategy_three(maze_init_,q, advanced_maze, see_future_step=1):
    '''
    using A star method
    :param maze_init:
    :param q:
    :return:
    '''
    start_point = [0, 0]
    dim = maze_init_.shape[0]
    maze_init = copy.deepcopy(maze_init_)
    current_step = 0
    while 1:
        advanced_maze_suppose = copy.deepcopy(maze_init)
        future_maze = generate_advanced_maze(maze_init, q=1, step=see_future_step)

        for new_fire_future in future_maze:
            for new_fire in  new_fire_future:
                advanced_maze_suppose[new_fire]=1
        future_maze.insert(0,[])
        shortest_path = Astar_search(advanced_maze_suppose, start_point)

        for pp in future_maze[::-1]:
            shortest_path = Astar_search(advanced_maze_suppose, start_point)
            if len(shortest_path)==0:
                for new_fire_future in pp:
                    advanced_maze_suppose[new_fire_future]=2
            else:
                break

        if len(shortest_path) == 0:
            return False
        next_point = shortest_path[1]
        # advanced_maze = generate_advanced_maze(maze_init,q ,step = 1)[0]
        for new_fire in advanced_maze[current_step]:
            maze_init[new_fire]=1

        if maze_init[next_point[0],next_point[1]] == 1:
            return False
        if next_point == [dim-1,dim-1]:
            return True
        start_point = next_point
        current_step+=1

def strategy_three(maze_init_,q, advanced_maze, see_future_step=1):
    '''
    using A star method
    :param maze_init:
    :param q:
    :return:
    '''
    start_point = [0, 0]
    dim = maze_init_.shape[0]
    maze_init = copy.deepcopy(maze_init_)
    current_step = 0
    while 1:
        advanced_maze_suppose = copy.deepcopy(maze_init)
        list_advanced_maze_suppose = []
        for new_fires in  generate_advanced_maze(maze_init, q=1, step=see_future_step):
            for new_fire in new_fires:
                advanced_maze_suppose[new_fire]=1
            list_advanced_maze_suppose.append(copy.deepcopy(advanced_maze_suppose))
        for cnt,supposed in enumerate(list_advanced_maze_suppose[::-1]):
            shortest_path = Astar_search(supposed, start_point)
            if len(shortest_path)==0:
                if cnt==(len(list_advanced_maze_suppose)-1):
                    shortest_path = Astar_search(maze_init, start_point)
                    if len(shortest_path) == 0:
                        return False
                    else:
                        break
                else:
                    continue
            else:
                break
        next_point = shortest_path[1]
        # advanced_maze = generate_advanced_maze(maze_init,q ,step = 1)[0]
        for new_fire in advanced_maze[current_step]:
            maze_init[new_fire]=1
        if maze_init[next_point[0],next_point[1]] == 1:
            return False
        if next_point == [dim-1,dim-1]:
            return True
        start_point = next_point
        current_step+=1

def compare_method(fn1,fn2,fn3,maze,q):
    dim = maze.shape[0]
    advanced_maze = generate_advanced_maze(maze,q,step=int(dim**2))
    l= [0,0,0,0]
    if fn1(maze, q, advanced_maze):
        l[0] += 1
    if fn2(maze, q, advanced_maze):
        l[1] += 1
    if fn3(maze, q, advanced_maze, see_future_step=1):
        l[2] += 1
    if fn3(maze, q, advanced_maze, see_future_step=4):
        l[3] += 1

    return l
@timer(True)
def plot_success_vs_flammability_multi_thread(fn1,fn2,fn3,dim=30,p=0.3,num_maze = 10, cnt_initial_fire = 10, use_multi_thread=False):
    q_list = [i for i in np.arange(0,1.01,0.05)]
    success_cnt1 = [0 for i in range(len(q_list))]
    success_cnt2 = [0 for i in range(len(q_list))]
    success_cnt3 = [0 for i in range(len(q_list))]
    success_cnt4 = [0 for i in range(len(q_list))]
    x_axis = [i*0.05 for i in range(len(q_list))]
    for _ in tqdm(range(num_maze)):
        maze,fire_source_list = generate_maze(dim,p,cnt_initial_fire)
        for fire_source in fire_source_list:
            maze[fire_source[0],fire_source[1]]=1
            if use_multi_thread:
                fn = partial(compare_method,fn1,fn2,fn3,maze)
                pool = Pool(len(q_list))
                result = pool.map(fn, q_list)
                pool.close()
                pool.join()
            else:
                result=[]
                for qq in q_list:
                    result.append(compare_method(fn1,fn2,fn3,maze,qq))
            for cnt,i in enumerate(result):
                success_cnt1[cnt] += i[0]
                success_cnt2[cnt] += i[1]
                success_cnt3[cnt] += i[2]
                success_cnt4[cnt] += i[3]
            maze[fire_source[0],fire_source[1]]=2
    success_rate1 = [1.0*i/num_maze/cnt_initial_fire for i in success_cnt1]
    success_rate2 = [1.0 * i / num_maze / cnt_initial_fire for i in success_cnt2]
    success_rate3 = [1.0 * i / num_maze / cnt_initial_fire for i in success_cnt3]
    success_rate4 = [1.0 * i / num_maze / cnt_initial_fire for i in success_cnt4]
    plt.plot(x_axis, success_rate1, label='strategy_1')
    plt.plot(x_axis, success_rate2, label='strategy_2')
    plt.plot(x_axis, success_rate3, label='strategy_3_n=1')
    plt.plot(x_axis, success_rate4, label='strategy_3_n=5')
    plt.title('average successes vs flammability q')
    plt.xlabel('flammability q')
    plt.ylabel('average successes rate')
    plt.legend()
    plt.savefig('multi_dim=%d.png'%(dim),dpi=500)



