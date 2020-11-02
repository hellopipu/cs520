firstly, you should install the package needed.

run the command

`pip install -r requirements.txt`

we use pygame to generate GUI in this project, white block means hidden cell, green block means safe cell revealed, red block means mine revealed, red block with X means mine we deduced.

- run the baseline agent

`python Minesweeper.py --name bsl --dim 5 --num 10 --wait_time 300`

- run the improved agent

`python Minesweeper.py --name bsl --dim 5 --num 10 --wait_time 300`

`dim` indicates dimension of the board, `num` indicates numbers of mine in the board, `wait_time` indicates waiting time for each round

- plot the score vs density curve

`python Minesweeper.py --dim 5 --plot`