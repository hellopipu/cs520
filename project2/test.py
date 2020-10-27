import matplotlib.pyplot as plt
from main import  *
import click
def debug_generate_random_env(dim = 20, mine=20):
    env = generate_random_env(dim,mine)
    plt.imshow(env)
    plt.show()

@click.command()
@click.option("--name",type=click.Choice(['gen','baseline','num_hidden']))
@click.option('--dim',default = 20)
@click.option('--n',default = 50)
def debug(name,dim,n):
    click.echo(name)
    if name=='gen':
        debug_generate_random_env(dim,n)
    elif name=='baseline':
        agent = Agent(dim,n)
        agent.strategy_BL()
    elif name=='num_hidden':
        KnowledgeBase(10)
debug()