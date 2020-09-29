from main import *
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from functools import partial
import time
from itertools import zip_longest

def debug_BFS_and_generate(dim=30):
    arr, fire_list = generate_maze(dim, 0.3)
    print(fire_list)

    i = 0
    temp = arr[fire_list[i][0], fire_list[i][1]]
    arr[fire_list[i][0], fire_list[i][1]] = 1
    start_point = [0, 0]
    path = BFS_search(arr, start_point)
    print('path: ',path)
    plt.subplot(121)
    plt.imshow(arr,'gray')
    for i in path:
        arr[i[0], i[1]] += 4
    plt.subplot(122)
    plt.imshow(arr,'gray')
    plt.show()
def debug_DFS_BFS_Astar_and_generate(dim=30):
    dfs_t = 0
    bfs_t = 0
    ast_t = 0
    repeat_nums = 100
    for _ in tqdm(range(repeat_nums)):
        arr, fire_source_list = generate_maze(dim, 0.3)
        for fire_source in fire_source_list:
            arr[fire_source[0], fire_source[1]] = 1
        arr2 = copy.deepcopy(arr)
        arr1 = copy.deepcopy(arr)
        arr0 = copy.deepcopy(arr)
        start_point = [0, 0]
        t0 = time.time()
        path0 = DFS_search(arr, [dim-1,dim-1])
        t1 = time.time()
        path1 = BFS_search(arr, start_point)
        t2 = time.time()
        path2 = Astar_search(arr, start_point)
        dfs_t += t1 - t0
        bfs_t += t2 - t1
        ast_t += time.time() - t2
    print('dfs time:',dfs_t,'bfs time:',bfs_t,'astar time: ',ast_t)
    print('path1: ', len(path0))
    print('path2: ', len(path1))
    print('path2: ', len(path2))
    plt.subplot(141)
    plt.imshow(arr,'gray')
    for k,i,j in zip_longest(path0,path1,path2,fillvalue=[0,0]):
        arr0[k[0], k[1]] = 6
        arr1[i[0], i[1]] = 6
        arr2[j[0], j[1]] = 6
    plt.subplot(142)
    plt.imshow(arr0, 'gray')
    plt.subplot(143)
    plt.imshow(arr1,'gray')
    plt.subplot(144)
    plt.imshow(arr2,'gray')
    plt.savefig('debug_DFS_BFS_Astar_and_generate.png',dpi=500)

def debug_generate_maze(dim=30):
    arr, fire_list = generate_maze(dim, 0.3)
    print('fire_point: ',fire_list)
################################ show ##########
    for i in range(10):
        plt.subplot(5,2,i+1)
        temp = arr[fire_list[i][0],fire_list[i][1]]
        arr[fire_list[i][0],fire_list[i][1]] = 1
        plt.imshow(arr,'gray')
        arr[fire_list[i][0],fire_list[i][1]] = temp
    plt.show()
################################ show ##########
def debug_advanced_maze(dim=30):
    arr, fire_source_list = generate_maze(dim, 0.3, cnt_initial_fire= 1)
    for fire_source in fire_source_list:
        arr[fire_source[0], fire_source[1]] = 1
    print('fire_point: ',fire_source_list)
    q = 1
    step = 10 #[i for i in range(0,50,5)]
    list_new_fire = generate_advanced_maze(arr, q, step)
    print(len(arr))
    print((list_new_fire))
    # for i in range(step):
    #
    # for cnt,i in enumerate(range(0,50,5)):
    #     plt.subplot(5,2,cnt+1)
    for cnt,fires in enumerate(list_new_fire):
        for f in fires:
            arr[f]=1
        if cnt%1==0:
            plt.subplot(5, 2, cnt//1+1)
            plt.imshow(arr,'gray')
    plt.savefig('debug_advanced_maze.png',dpi=500)

def debug_strategy_one(dim=30):
    arr, fire_list = generate_maze(dim, 0.3)
    i = 0
    temp = arr[fire_list[i][0], fire_list[i][1]]
    arr[fire_list[i][0], fire_list[i][1]] = 1
    q = 0.1
    return strategy_one(arr, q)
def debug_cost_map(dim=20,q=0.3):
    arr, fire_source_list = generate_maze(dim, 0.3)
    for fire_source in fire_source_list:
        arr[fire_source[0], fire_source[1]] = 1
    cost_map = generate_cost_map(arr,fire_source_list,q)
    plt.subplot(121)
    plt.imshow(arr)
    plt.subplot(122)
    plt.imshow(cost_map)
    plt.show()

# debug_BFS_and_generate()
# debug_DFS_BFS_Astar_and_generate(20)
# debug_generate_maze()
# debug_advanced_maze()
# print(debug_strategy_one())
# debug_cost_map()

# plot_success_vs_flammability(strategy_one,dim=10,p=0.3,num_maze = 10, cnt_initial_fire = 10)
# plot_success_vs_flammability(strategy_one,strategy_two,dim=5,p=0.3,num_maze = 10, cnt_initial_fire = 10)
# plot_success_vs_flammability(strategy_two,strategy_three,dim=5,p=0.3,num_maze = 10, cnt_initial_fire = 10)

plot_method = partial(plot_success_vs_flammability_multi_process ,dim=30,p=0.3,num_maze = 10, cnt_initial_fire = 10)
plot_method(strategy_one,strategy_two,strategy_three)