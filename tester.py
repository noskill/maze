import pygame
from maze import Maze
from robot import Robot
from image import Image, DummyImage
import threading
import logging
import os
import argparse
import sys
import random
import math
import tempfile

from genmaze import generate_maze, write_maze

# global dictionaries for robot movement and sensing
dir_sensors = {'u': ['l', 'u', 'r'], 'r': ['u', 'r', 'd'],
               'd': ['r', 'd', 'l'], 'l': ['d', 'l', 'u'],
               'up': ['l', 'u', 'r'], 'right': ['u', 'r', 'd'],
               'down': ['r', 'd', 'l'], 'left': ['d', 'l', 'u']}
dir_move = {'u': [0, 1], 'r': [1, 0], 'd': [0, -1], 'l': [-1, 0],
            'up': [0, 1], 'right': [1, 0], 'down': [0, -1], 'left': [-1, 0]}
dir_reverse = {'u': 'd', 'r': 'l', 'd': 'u', 'l': 'r',
               'up': 'd', 'right': 'l', 'down': 'u', 'left': 'r'}

# test and score parameters
max_time = 1000
train_score_mult = 1/30.

done = False
game_objects = []
logger = logging.getLogger(__name__)

def configure_logging(args):
    level = os.environ.get("LOG_LEVEL", "").upper()
    if args.debug:
        level = "DEBUG"
    elif args.log:
        level = args.log.upper()

    if not level:
        log_level = logging.WARNING
    else:
        log_level = getattr(logging, level, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(levelname)s:%(name)s:%(message)s",
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Test robot on a maze.")
    parser.add_argument("maze", help="Path to maze file or 'random'.")
    parser.add_argument("--gui", action="store_true", help="Enable visualization.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--log", default="", help="Set log level (e.g. INFO, DEBUG).")
    parser.add_argument("--save-path", default="", help="Save run data to file.")
    parser.add_argument("-N", dest="num_mazes", type=int, default=1, help="Number of mazes to run.")
    return parser.parse_args()

def pygame_loop():
    clock = pygame.time.Clock()
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        ms = clock.tick(30)
        for o in game_objects:
            o.update(ms)
        pygame.display.update()

def random_even_dim(rng):
    return rng.randrange(12, 26, 2)


def random_extra_openings(dim, rng):
    low = int(math.floor(dim ** 1.5))
    high = int(math.ceil(dim ** 1.8))
    return rng.randint(low, high)


def generate_random_maze_file(rng):
    dim = random_even_dim(rng)
    extra_openings = random_extra_openings(dim, rng)
    seed = rng.randrange(1 << 31)
    walls = generate_maze(dim, seed=seed, extra_openings=extra_openings)
    name = "random_maze_{}_{}_{}.txt".format(dim, extra_openings, seed)
    print(name)
    path = os.path.join(tempfile.gettempdir(), name)
    write_maze(path, dim, walls)
    logger.info(
        "Generated random maze %s (dim=%s extra_openings=%s seed=%s)",
        path,
        dim,
        extra_openings,
        seed,
    )
    map_id = "random dim={} extra_openings={} seed={}".format(dim, extra_openings, seed)
    return path, map_id


def run_single_maze(maze_path, map_id, args, data_file):
    '''
    This script tests a robot based on the code in robot.py on a maze given
    as an argument when running the script.
    '''

    global done, game_objects
    done = False
    game_objects = []

    gui = args.gui
    # Create a maze based on input argument on command line.
    testmaze = Maze(str(maze_path))
    if gui:
        pygame.init()
        size = 50 * testmaze.dim + 100
        screen = pygame.display.set_mode((size, size))
        image = Image(testmaze.dim, screen)
        thread = threading.Thread(target=pygame_loop)
        thread.daemon = True
        thread.start()
    else:
        image = DummyImage()
    # Intitialize a robot; robot receives info about maze dimensions.
    game_objects.append(image)
    testrobot = Robot(testmaze.dim, image)
    
    
    # Record robot performance over two runs.
    runtimes = []
    total_time = 0
    times_to_run = 2
    if data_file is not None:
        times_to_run = 1
    for run in range(times_to_run):
        print("Starting run {}.".format(run))
        if run == 0:
            if data_file is not None:
                data_file.write('\n' + str(map_id) + '\n')

        # Set the robot in the start position. Note that robot position
        # parameters are independent of the robot itself.
        robot_pos = {'location': [0, 0], 'heading': 'up'}

        run_active = True
        hit_goal = False
        while run_active:
            total_time += 1
            # check for end of time
            if total_time > max_time:
                run_active = False
                print ("Allotted time exceeded.")
                break

            # provide robot with sensor information, get actions
            sensing = [testmaze.dist_to_wall(robot_pos['location'], heading)
                       for heading in dir_sensors[robot_pos['heading']]]
            rotation, movement = testrobot.next_move(sensing)

            # check for a reset
            if (rotation, movement) == ('Reset', 'Reset'):
                if run == 0 and hit_goal:
                    run_active = False
                    runtimes.append(total_time)
                    print ("Ending first run. Starting next run.")
                    break
                elif run == 0 and not hit_goal:
                    print ("Cannot reset - robot has not hit goal yet.")
                    continue
                else:
                    print ("Cannot reset on runs after the first.")
                    continue

            if run == 0 and data_file is not None:
                data = dict()
                data.update(robot_pos)
                data['sensor'] = sensing
                data['action'] = rotation, movement
                data_file.write(str(data) + ';')
                data_file.flush()

            # perform rotation
            if rotation == -90:
                robot_pos['heading'] = dir_sensors[robot_pos['heading']][0]
            elif rotation == 90:
                robot_pos['heading'] = dir_sensors[robot_pos['heading']][2]
            elif rotation == 0:
                pass
            else:
                print ("Invalid rotation value, no rotation performed.")

            # perform movement
            if abs(movement) > 3:
                print ("Movement limited to three squares in a turn.")
            movement = max(min(int(movement), 3), -3) # fix to range [-3, 3]
            while movement:
                if movement > 0:
                    if testmaze.is_permissible(robot_pos['location'], robot_pos['heading']):
                        robot_pos['location'][0] += dir_move[robot_pos['heading']][0]
                        robot_pos['location'][1] += dir_move[robot_pos['heading']][1]
                        movement -= 1
                    else:
                        print ("Movement stopped by wall.")
                        movement = 0
                else:
                    rev_heading = dir_reverse[robot_pos['heading']]
                    if testmaze.is_permissible(robot_pos['location'], rev_heading):
                        robot_pos['location'][0] += dir_move[rev_heading][0]
                        robot_pos['location'][1] += dir_move[rev_heading][1]
                        movement += 1
                    else:
                        print ("Movement stopped by wall.")
                        movement = 0

            # check for goal entered
            goal_bounds = [testmaze.dim/2 - 1, testmaze.dim/2]
            if robot_pos['location'][0] in goal_bounds and robot_pos['location'][1] in goal_bounds:
                hit_goal = True
                if run != 0:
                    runtimes.append(total_time - sum(runtimes))
                    run_active = False
                    print ("Goal found; run {} completed!".format(run))

    # Report score if robot is successful.
    if len(runtimes) == 2:
        print ("Task complete! Score: {:4.3f}".format(runtimes[1] + train_score_mult*runtimes[0]))
    done = True
    if gui:
        thread.join()
        pygame.quit()


def run(args):
    rng = random.Random()
    if args.maze == "random":
        maze_items = [generate_random_maze_file(rng) for _ in range(args.num_mazes)]
    else:
        maze_items = [(args.maze, args.maze) for _ in range(args.num_mazes)]

    data_file = None
    if args.save_path:
        data_file = open(args.save_path, 'at')

    try:
        i = 0
        for maze_path, map_id in maze_items:
            print(i)
            run_single_maze(maze_path, map_id, args, data_file)
            if args.maze == 'random':
                os.remove(maze_path)
            i += 1
    finally:
        if data_file is not None:
            data_file.close()


if __name__ == '__main__':
    try:
        args = parse_args()
        configure_logging(args)
        run(args)
    except KeyboardInterrupt:
        done = True
        sys.exit(0)
