# Maze solver

dependencies:
pygame
numpy


Usage:
python tester.py test_maze_01.txt

or with vizualisation:

python tester.py test_maze_01.txt --gui

To show map run:

python showmaze.py test_maze_01.txt

Generate a random maze:

python genmaze.py 12 random_maze_12.txt

to generate dataset on random mazes:

python tester.py random -N 500 --log=WARN --save-path data.txt
