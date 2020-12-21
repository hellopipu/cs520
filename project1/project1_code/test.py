from main import *
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import time
from itertools import zip_longest


def debug_BFS_and_generate(dim=30):
    arr, fire_list = generate_maze(dim, 0.3)
    i = 0
    arr[fire_list[i][0], fire_list[i][1]] = 1
    start_point = [0, 0]
    path = BFS_search(arr, start_point)
    plt.subplot(121)
    plt.imshow(arr, 'gray')
    for i in path:
        arr[i[0], i[1]] += 4
    plt.subplot(122)
    plt.imshow(arr, 'gray')
    plt.savefig('debug_BFS_and_generate.png',dpi=500)


def debug_DFS_BFS_Astar_and_generate(dim=30, repeat_nums=10):
    dfs_t = 0
    bfs_t = 0
    ast_t = 0

    for _ in tqdm(range(repeat_nums)):
        arr, fire_source_list = generate_maze(dim, 0.3, 1)
        for fire_source in fire_source_list:
            arr[fire_source[0], fire_source[1]] = 1
        arr2 = copy.deepcopy(arr)
        arr1 = copy.deepcopy(arr)
        arr0 = copy.deepcopy(arr)
        start_point = [0, 0]
        t0 = time.time()
        path0 = DFS_search(arr, [dim - 1, dim - 1])
        t1 = time.time()
        path1 = BFS_search(arr, start_point)
        t2 = time.time()
        path2 = Astar_search(arr, start_point)
        dfs_t += t1 - t0
        bfs_t += t2 - t1
        ast_t += time.time() - t2
    print('dfs time:', dfs_t / repeat_nums * 1000, 'bfs time:', bfs_t / repeat_nums * 1000, 'astar time: ',
          ast_t / repeat_nums * 1000)

    plt.subplot(141)
    plt.imshow(arr, 'gray')
    plt.title('maze'), plt.xticks([]), plt.yticks([])
    for k, i, j in zip_longest(path0, path1, path2, fillvalue=[0, 0]):
        arr0[k[0], k[1]] = 6
        arr1[i[0], i[1]] = 6
        arr2[j[0], j[1]] = 6
    plt.subplot(142)
    plt.imshow(arr0, 'gray')
    plt.title('DFS path'), plt.xticks([]), plt.yticks([])
    plt.subplot(143)
    plt.imshow(arr1, 'gray')
    plt.title('BFS path'), plt.xticks([]), plt.yticks([])
    plt.subplot(144)
    plt.imshow(arr2, 'gray')
    plt.title('Astar Path'), plt.xticks([]), plt.yticks([])
    plt.savefig('debug_DFS_BFS_Astar_and_generate.png', dpi=500)


def debug_generate_maze(repeat_num=100):
    arr1, fire_list = generate_maze(5, p=0.3, cnt_initial_fire=1)
    arr1[fire_list[0][0], fire_list[0][1]] = 1

    arr2, fire_list = generate_maze(10, p=0.3, cnt_initial_fire=1)
    arr2[fire_list[0][0], fire_list[0][1]] = 1

    arr3, fire_list = generate_maze(20, p=0.3, cnt_initial_fire=1)
    arr3[fire_list[0][0], fire_list[0][1]] = 1

    arr4, fire_list = generate_maze(50, p=0.3, cnt_initial_fire=1)
    arr4[fire_list[0][0], fire_list[0][1]] = 1
    plt.subplot(141)
    plt.title('dim=5')
    plt.xticks([]), plt.yticks([])
    plt.imshow(arr1, 'gray')
    plt.subplot(142)
    plt.title('dim=10')
    plt.xticks([]), plt.yticks([])
    plt.imshow(arr2, 'gray')
    plt.subplot(143)
    plt.title('dim=20')
    plt.xticks([]), plt.yticks([])
    plt.imshow(arr3, 'gray')
    plt.subplot(144)
    plt.title('dim=50')
    plt.xticks([]), plt.yticks([])
    plt.imshow(arr4, 'gray')
    plt.savefig('sample_mazes.png', dpi=500)

    t0 = time.time()
    for _ in tqdm(range(repeat_num)):
        generate_maze(5, p=0.3, cnt_initial_fire=1)
    t1 = time.time()
    for _ in tqdm(range(repeat_num)):
        generate_maze(10, p=0.3, cnt_initial_fire=1)
    t2 = time.time()
    for _ in tqdm(range(repeat_num)):
        generate_maze(20, p=0.3, cnt_initial_fire=1)
    t3 = time.time()
    for _ in tqdm(range(repeat_num)):
        generate_maze(50, p=0.3, cnt_initial_fire=1)
    t4 = time.time()

    print('testing the time cost of different dimesions when generating mazes:')
    print('dim = 5: %.2f ms' % ((t1 - t0) * 10))
    print('dim = 10: %.2f ms' % ((t2 - t1) * 10))
    print('dim = 20: %.2f ms' % ((t3 - t2) * 10))
    print('dim = 50: %.2f ms' % ((t4 - t3) * 10))


def debug_advanced_maze(dim=20):
    arr, fire_source_list = generate_maze(dim, 0.3, cnt_initial_fire=1)
    for fire_source in fire_source_list:
        arr[fire_source[0], fire_source[1]] = 1
    q = 1
    step = 50  # [i for i in range(0,50,5)]
    list_new_fire = generate_advanced_maze(arr, q, step)
    plt.subplot(2, 5, 1)
    plt.title('step=0')
    plt.xticks([]), plt.yticks([])
    plt.imshow(arr, 'gray')
    for cnt, fires in enumerate(list_new_fire):
        for f in fires:
            arr[f] = 1
        if cnt % 5 == 0:
            if cnt // 5 + 2 > 10:
                break
            plt.subplot(2, 5, cnt // 5 + 2)
            plt.title('step=%d' % (cnt))
            plt.xticks([]), plt.yticks([])
            plt.imshow(arr, 'gray')
    plt.savefig('debug_advanced_maze.png', dpi=500)


def debug_cost_map(dim=20, q=0.3):
    arr, fire_source_list = generate_maze(dim, 0.3, 5)
    for fire_source in fire_source_list:
        arr[fire_source[0], fire_source[1]] = 1
    cost_map = generate_cost_map(arr, fire_source_list, q)

    plt.subplot(121)
    plt.imshow(arr, 'gray')
    plt.title('maze'), plt.xticks([]), plt.yticks([])
    plt.subplot(122)
    plt.imshow(cost_map)
    plt.title('heatmap'), plt.xticks([]), plt.yticks([])
    plt.savefig('heatmap.png', dpi=500)


def compare_startegy_time_cost(dim=10, p=0.3, q=0.5, num_maze=10, cnt_initial_fire=1):
    s1 = 0
    s2 = 0
    s3 = 0
    for _ in tqdm(range(num_maze)):
        maze, fire_source_list = generate_maze(dim, p, cnt_initial_fire)
        for fire_source in fire_source_list:
            maze[fire_source[0], fire_source[1]] = 1
        advanced_maze = generate_advanced_maze(maze, q, step=int(dim ** 2))
        t0 = time.time()
        strategy_one(maze, advanced_maze)
        t1 = time.time()
        strategy_two(maze, advanced_maze)
        t2 = time.time()
        strategy_three(maze, advanced_maze)
        t3 = time.time()
        s1 += t1 - t0
        s2 += t2 - t1
        s3 += t3 - t2
    print('strategy 1: %.2f ms' % (s1 / num_maze * 1000.))
    print('strategy 2: %.2f ms' % (s2 / num_maze * 1000.))
    print('strategy 3: %.2f ms' % (s3 / num_maze * 1000.))


if __name__ == '__main__':
    # debug_BFS_and_generate()
    # debug_DFS_BFS_Astar_and_generate(dim=20,repeat_nums=10)
    # debug_generate_maze()
    # debug_advanced_maze()
    # debug_cost_map(q=1)
    # compare_startegy_time_cost(dim = 5)

    #### plot_success_vs_flammability
    plot_method = partial(plot_success_vs_flammability_multi_thread, dim=5, p=0.3, num_maze=10, cnt_initial_fire=10,
                          use_multi_thread=False)
    plot_method(strategy_one, strategy_two, strategy_three)
