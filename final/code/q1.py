import numpy as np
import random
import matplotlib.pyplot as plt
random.seed(9)
def cal_mahanttan(sheep,dog):
    return abs(sheep[0]-dog[0]) + abs(sheep[1]-dog[1])
def isCatch(maze,sheep):
    if maze[sheep[0]-1,sheep[1]]==1 and maze[sheep[0], sheep[1]-1] == 1:
        return True
    else:
        return False
def get_move_pos(pos,other_pos, sheep,size,det, first_move=True):
    '''
        dog move
    :param pos: dog position
    :param maze:
    :param det:  destilation (left and top)
    :param first_move:
    :return:
    '''

    l=[]
    for i in range(-1,2):
        if 0<=pos[0]+i<size and (pos[0]+i,pos[1])!=sheep :
            if first_move or (pos[0]+i,pos[1])!=other_pos:
                l.append((pos[0]+i,pos[1]))
    for j in range(-1,2):
        if 0<=pos[1]+j<size and (pos[0],pos[1]+j)!=sheep :
            if first_move or (pos[0],pos[1]+j)!=other_pos:
                l.append((pos[0],pos[1]+j))
    min_manh = size*2

    for ll in l:
        dist = cal_mahanttan(det,ll)
        if dist<min_manh:
            min_manh = dist
            min_pos = ll
    return min_pos

def random_move(pos,dog1,dog2):
    '''
    sheep mpve
    :param maze:
    :param pos:
    :return:
    '''
    l=[]
    size = maze.shape[0]
    for i in [-1,1]:
        if 0 <= pos[0] + i < size  and (pos[0] + i, pos[1]) != dog1 and (pos[0] + i, pos[1]) != dog2:
            l.append((pos[0] + i,pos[1]))
    for j in [-1,1]:
        if 0 <= pos[1] + j < size  and (pos[0], pos[1]+j) != dog1 and (pos[0], pos[1]+j) != dog2:
            l.append((pos[0],pos[1]+j))
    if len(l)!=0:
        return random.choice(l)
    else:
        return pos
def update_maze(dog1,dog2,sheep,size):
    maze = np.zeros((size, size))
    maze[sheep] = 2
    maze[dog1] = 1
    maze[dog2] = 1
    return maze



Repeat = 1000
r = 0
c = 0
while(r<Repeat):
    r+=1
    size = 7
    x = random.sample(range(0, 7 * 7), 3)
    ##
    x = [28, 6, 45]
    sheep = (x[0] // size, x[0] % size)
    dog1 = (x[1] // size, x[1] % size)
    dog2 = (x[2] // size, x[2] % size)
    maze = np.zeros((size, size))
    maze[sheep] = 2
    maze[dog1] = 1
    maze[dog2] = 1

    cnt = 0
    # print('cnt: ', cnt, 'dog: ', dog1, dog2, 'sheep: ', sheep)
    flag = isCatch(maze, sheep)
    # plt.imshow(maze)
    # plt.show()
    while not flag:
        cnt += 1
        dog1 = get_move_pos(dog1, dog2, sheep, size, (max(sheep[0] - 1, 0), sheep[1]))
        dog2 = get_move_pos(dog2, dog1, sheep, size, (sheep[0], (max(sheep[1] - 1, 0))), False)
        sheep = random_move(sheep, dog1, dog2)
        maze = update_maze(dog1, dog2, sheep, size)
        # print('cnt: ', cnt, 'dog: ', dog1, dog2, 'sheep: ', sheep)
        # plt.imshow(maze)
        # plt.show()
        flag = isCatch(maze, sheep)
        ## if in the other corner
        if sheep == (0, 0) and maze[0, 1] == 1 and maze[1, 0] == 1:
            cnt += 2 * size - 1
            break
        elif sheep == (0, size - 1) and maze[0, size - 2] == 1 and maze[1, size - 1] == 1:
            cnt += size + 1
            break
        elif sheep == (size - 1, 0) and maze[size - 2, 0] == 1 and maze[size - 1, 1] == 1:
            cnt += size + 1
            break
    c+=cnt
print('finish round: ', c/Repeat)






