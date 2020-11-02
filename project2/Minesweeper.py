import sys
import pygame
import click
from pygame.locals import *
from solver_bsl import Environment
from solver_bsl import Agent as Agent_bsl
from solver_imp import Agent as Agent_imp
import matplotlib.pyplot as plt
import numpy as np
import  random
seed = 10
random.seed(seed)
np.random.seed(seed)

# color
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
GREEN = (0, 255, 127)
BROWN = (139, 69, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


def draw_grid(ROWS,WIDTH,win):
    gap = WIDTH // ROWS

    ## starting points
    x=0
    y=0
    for i in range(ROWS):
        x=i*gap
        pygame.draw.line(win,GRAY,(x,0),(x,WIDTH),1)
        pygame.draw.line(win, GRAY, (0, x), (WIDTH,x), 1)

def render(agent ,WIDTH,win,END_FONT):
    win.fill(WHITE)
    draw_grid(agent.dim,WIDTH,win)
    width_block = WIDTH/agent.dim
    for cell in agent.cells_visited:
        i,j = cell
        if cell not in agent.safes:
            win.fill(RED, pygame.Rect(j*width_block+2, i*(width_block)+2, width_block-4, width_block-4))
        else:
            win.fill(GREEN, pygame.Rect(j * width_block+2, i * (width_block)+2, width_block-4, width_block-4))
            line = END_FONT.render(str(int(agent.cells_info_from_env[cell])), True, BLACK)
            lineRect = line.get_rect()
            lineRect.center = (j * width_block+ width_block/2 , i * width_block + width_block/2 )
            win.blit(line, lineRect)
    for cell in agent.mines - agent.cells_visited:
        i,j = cell
        # print(cell,'xxxxx')

        win.fill(RED, pygame.Rect(j*width_block+2, i*(width_block)+2, width_block-4, width_block-4))
        line = END_FONT.render("X", True, BLACK)
        lineRect = line.get_rect()
        lineRect.center = (j * width_block+ width_block/2 , i * width_block + width_block/2 )
        win.blit(line, lineRect)
    pygame.display.update()
    # return len(agent.mines),len(agent.mines - agent.cells_visited)

@click.command()
@click.option("--name",default = 'imp',type=click.Choice(['bsl','imp']))
@click.option('--dim',default = 5,type = int)
@click.option('--num',default = 10,type = int)
@click.option('--wait_time',default = 500,type = int)
@click.option('--plot',is_flag=True)

def main(name,dim,num,wait_time,plot ):

    if plot:
        print('plot score vs mine_density')
        density = [i*0.1 for i in range(1,11)]
        score_bsl = [0 for _ in range(1, 11)]
        score_imp = [0 for _ in range(1, 11)]
        num_list = [int(dim*dim*d) for d in density]
        num_repeat = 10
        for index,nn in enumerate(num_list):
            for _ in range(num_repeat):
                env = Environment(dim, nn)
                for agent,score in zip([Agent_bsl(dim),Agent_imp(dim)],[score_bsl,score_imp]):
                    print(agent,'````````')
                    while True:
                        cell_be_explore = agent.explore()
                        if cell_be_explore == None:
                            score[index] += agent.score(nn)
                            print('index',agent.score(nn))
                            break
                        else:
                            cell_count = env.report(cell_be_explore)
                            agent.strategy(cell_be_explore, cell_count)
        score_bsl = [i/num_repeat for i in score_bsl]
        score_imp = [i / num_repeat for i in score_imp]
        plt.plot(density, score_bsl,'r',label='baseline')
        plt.plot(density, score_imp, 'b',label='improved')
        plt.legend()
        plt.xlabel('density')
        plt.ylabel('score')
        plt.title("score vs mine density, dim=%d"%dim)
        plt.savefig("plot_%d.png"%dim  ,dpi=300)
    else:
        env = Environment(dim, num)
        if name == 'bsl':
            agent = Agent_bsl(dim)
        elif name == 'imp':
            agent = Agent_imp(dim)
        pygame.init()
        # cell size in screen
        size =  30 if dim<30 else 1000//dim #30 if dim<10 else 20 if dim<50 else 15 if dim<60 else 13 if dim <70 else 10
        # board size
        WIDTH = size*dim

        win = pygame.display.set_mode((WIDTH, WIDTH))
        pygame.display.set_caption("MineSweeper")

        # font
        END_FONT = pygame.font.SysFont('courier', size)

        win.fill(WHITE)
        draw_grid(dim,WIDTH,win)
        round = 0
        game_finished = 0
        while True:
            if not game_finished:
                print('round: ',round)
                render(agent,WIDTH,win,END_FONT)
                cell_be_explore = agent.explore()
                if cell_be_explore == None:
                    print("finish game")
                    print("final score: ", agent.score(num))
                    game_finished = 1
                else:
                    cell_count = env.report(cell_be_explore)
                    agent.strategy(cell_be_explore,cell_count)
                    round+=1
                if 0: # save every round imgs
                    pygame.image.save(win,"progress/%d.png"%round)
                pygame.time.wait(wait_time)
            else:
                render(agent, WIDTH, win, END_FONT)
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
main()