import sys
import astar
from math import pi
import numpy
from mouse_map import Map
import logging


logger = logging.getLogger()
np = numpy

def angle(v1, v2):
    return numpy.arctan2(v2[1], v2[0]) - numpy.arctan2(v1[1], v1[0])


def tuplify(arr):
    if arr.ndim < 2:
        return tuple(x for x in arr)
    return tuple(tuplify(x) for x in arr)


class Robot(object):
    def __init__(self, maze_dim, image):
        '''
        Use the initialization function to set up attributes that your robot
        will use to learn and navigate the maze. Some initial attributes are
        provided based on common information, including the size of the maze
        the robot is placed in.
        '''

        self.location = np.asarray([0, 0])
        self.heading = 'up'
        self.maze_dim = maze_dim
        self.goal = maze_dim/2 - 1, maze_dim/2
        self.map = Map(maze_dim, maze_dim)
        self.moves = (0, 1), (1,0), (0, -1), (-1, 0)
        self.image = image
        for i in range(self.maze_dim):
            for j in range(self.maze_dim):
                for x,y in self.moves:
                    x_new, y_new = i + x, j + y
                    if self.is_valid(x_new, y_new):
                        self.map.add_edge((i, j), (x_new, y_new))
                        self.map[(i,j)] = abs(self.goal[0] - i) + abs(self.goal[1] - j)
                        self.image.update_text((i, j), self.map[i, j])

        self._pos = numpy.asarray([0,0])
        self._ori = numpy.asarray([[1, 0],[0, 1]])
        self.RLeft = numpy.asarray([[0, -1],[1, 0]])
        self.RRight = numpy.asarray([[0, 1],[-1, 0]])

        self.sq_size = 40
        self.draw_orig = numpy.asarray([(self.maze_dim * self.sq_size / -2) for x in range(2)])
        self.visited = set()
        # list of ((pos, orientation), action)) pairs
        self.path = None
        self.to_explore = []
        self.run = 0
        self.call_on_map_change = None
        self.runtime = [0, 0, 0]

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value
        self.image.move(value)

    @property
    def ori(self):
        return self._ori

    @ori.setter
    def ori(self, value):
        ang = angle(numpy.dot(self.ori, [0,1]), numpy.dot(value, [0, 1]))
        self.image.rotate(ang)
        self._ori = value


    def is_valid(self, x_new, y_new):
        if 0 <= x_new < self.maze_dim and\
           0 <= y_new < self.maze_dim:
               return True
        return False

    def next_move(self, sensors):
        '''
        Use this function to determine the next move the robot should make,
        based on the input from the sensors after its previous move. Sensor
        inputs are a list of three distances from the robot's left, front, and
        right-facing sensors, in that order.

        Outputs should be a tuple of two values. The first value indicates
        robot rotation (if any), as a number: 0 for no rotation, +90 for a
        90-degree rotation clockwise, and -90 for a 90-degree rotation
        counterclockwise. Other values will result in no rotation. The second
        value indicates robot movement, and the robot will attempt to move the
        number of indicated squares: a positive number indicates forwards
        movement, while a negative number indicates backwards movement. The
        robot may move a maximum of three units per turn. Any excess movement
        is ignored.

        If the robot wants to end a run (e.g. during the first training run in
        the maze) then returing the tuple ('Reset', 'Reset') will indicate to
        the tester to end the run and return the robot to the start.
        '''

        self.visited.add(tuple(self.pos))
        if self.run == 0:
            result = self.zero_run(sensors)
        elif self.run == 1:
            result = self.explore_run(sensors)
        else:
            result = self.final_run(sensors)
            print "Runtime: ", self.runtime
        if result == ("Reset", "Reset"):
            return result
        self.runtime[self.run] += 1
        ang = result[0]
        if ang < 0:
            self.ori = numpy.dot(self.RLeft, self.ori)
            logger.info("rotate left")
        if 0 < ang:
            self.ori = numpy.dot(self.RRight, self.ori)
            logger.info("rotate right")

        if result[1] != 0:
            self.pos = numpy.dot(self.ori, [0, result[1]]) + self.pos
            logger.info("move forward")

        return result

    def explore_run(self, sensors):
        """
        Explore potentially optimal paths from start position to the goal
        using a-star algorithm
        """
        self.process_sensors(sensors)
        def on_map_change():
            initial = ((0, 0), ((1, 0), (0,1)))
            is_goal = lambda ((x, y), ori): self.is_goal(x, y)
            #heuristic = lambda (pos, ori): sum(numpy.abs(np.asarray(pos) - self.pos)) / 3.0
            heuristic = lambda (pos, ori): self.map[pos] / 3.0
            actions_with_turn = lambda (pos, ori): self.possible_actions(pos, ori, rotate_and_move=True)
            self.path = astar.search(initial, is_goal, heuristic, actions_with_turn)
            logger.info("New path: ", self.path)
            self.to_explore = []
            for item in self.path:
                pos = item[1][0]
                if not(pos in self.visited or pos in self.to_explore):
                    self.to_explore.append(pos)
        if self.path is None:
            on_map_change()
        self.call_on_map_change = on_map_change
        logger.info("To explore: ", self.to_explore)
        logger.info("Pos: ", self.pos)

        if len(self.to_explore) and all(self.pos == self.to_explore[-1]):
            self.to_explore.pop(-1)

        if len(self.to_explore) == 0:
            self.run += 1
            self.reset()
            return 'Reset', 'Reset'

        next_point = self.to_explore[-1]
        initial = tuplify(self.pos), tuplify(self.ori)
        is_goal = lambda ((x, y), ori), target=next_point: x == target[0] and y == target[1]
        heuristic = lambda (pos, ori): sum(numpy.abs(np.asarray(pos) - self.pos)) / 3.0
        actions = lambda (pos, ori): self.possible_actions(pos, ori)
        path = astar.search(initial, is_goal, heuristic, actions)
        return path[-1][0]

    def reset(self):
        """
        Set position and orientation to initial values
        """
        self._pos = numpy.asarray([0,0])
        self._ori = numpy.asarray([[1, 0],[0, 1]])
        self.image.reset_arrow()

    def zero_run(self, sensors):
        """
        compute next movement using flood fill algorithm
        """
        x, y = self.pos
        if self.is_goal(x, y):
            self.run += 1
            return self.explore_run(sensors)
        # process sensors
        self.process_sensors(sensors)
        return self.flood_fill_next_step()

    def final_run(self, sensors):
        """
        Returns actions from exploration run
        """
        next_item = self.path.pop(-1)
        return next_item[0]

    def possible_actions(self, pos, ori, rotate_and_move=False):
        """
        Returns all (state, action to achive state) pairs directly achiveable given robot's position
        and orientation
        -------
        pos: numpy.array 1x2 or tuple
            robot's position
        ori: numpy.array 3x3 or tuple
             robot's orientation matrix
        rotate_and_move: bool
            if true allow rotate and move actions

        Returns
        -------
        list of ((position, orientation), action) pairs
        """

        # rotate left
        result = [((pos, tuplify(numpy.dot(self.RLeft, ori))), (-90, 0))]
        # rotate right
        result.append(((pos, tuplify(numpy.dot(self.RRight, ori))), (90, 0)))
        # strait
        t_range = [(0, numpy.asarray(ori))]
        if rotate_and_move:
            # left
            t_range.append((-90, numpy.dot(self.RLeft, ori)))
            # right
            t_range.append((90, numpy.dot(self.RRight, ori)))
        for (angle, t_ori) in t_range:
            t_pos = np.asarray(pos)
            for i in range(1, 4):
                new_pos = t_pos + numpy.dot(np.asarray(t_ori), [0, 1])
                if self.is_valid(*new_pos) and self.map.is_connected(t_pos, new_pos):
                    result.append(((tuplify(new_pos), tuplify(t_ori)), (angle, i)))
                else:
                    break
                t_pos = new_pos
        return result

    def get_rotation(self, pos, ori, target_pos):
        """
        returns angle to rotate towards target position in radians
        -------
        pos: numpy.array 1x2 or tuple
        position
        ori: numpy.array 3x3 or tuple
        current orientation
        target_pos: numpy.array 1x2 or tuple
        postion to rotate to
        -------
        returns angle in radians, clockwise rotatation has negative sign
        """
        ang =  angle(numpy.dot(ori, [0,1]), target_pos - pos)
        if pi < abs(ang):
            ang = ang % pi * -ang/(abs(ang))
        return ang

    def is_goal(self, x, y):
        if x in self.goal and y in self.goal:
            return True
        return False

    def process_sensors(self, a_sensors):
        """
        Checks if new walls were found, if so updates the map
        """
        # +1 becouse, there is nothing to rotate if sensors is zero
        sensors = [(u + 1) for u in a_sensors]

        # get vectors
        left = numpy.dot(self.RLeft, [0, sensors[0]])
        strait = numpy.asarray([0, sensors[1]])
        right = numpy.dot(self.RRight, [0, sensors[2]])

        # rotate sensors
        m_left = numpy.dot(self.ori, left)
        m_strait = numpy.dot(self.ori, strait)
        m_right = numpy.dot(self.ori, right)


        substract = lambda w: w + [(x/abs(x) * -1 if x else 0) for x in w]

        for m_to in [m_left, m_strait, m_right]:
            # there is wall between these positions
            m_from = self.pos + substract(m_to)
            m_to += self.pos
            if self.is_valid(*m_to):
                if self.map.is_connected(m_from, m_to):
                    logger.info("Remove edge {0} to {1}".format(m_from, m_to))
                    self.map.remove_edge(m_from, m_to)
                    self.update_map_from(m_to)
                    self.update_map_from(m_from)
                    self.image.draw_line_between(m_from, m_to)
                    if self.call_on_map_change is not None:
                        self.call_on_map_change()


    def get_candidate_pos(self, a_pos):
        candidate_pos = []
        for m in self.moves:
            neib = a_pos + m
            if self.is_valid(*neib) and self.map.is_connected(a_pos, neib):
                candidate_pos.append(neib)
        return candidate_pos

    def flood_fill_next_step(self):
        """
        compute next movement using flood fill algorithm
        """
        candidate_pos = self.get_candidate_pos(self.pos)
        best = candidate_pos[0], self.map[candidate_pos[0]]
        candidate_pos.pop(0)
        for p in candidate_pos:
            dist = self.map[p]
            if dist < best[1]:
                best = p, dist
            elif dist == best[1]:
                if abs(self.get_rotation(self.pos, self.ori, p)) < abs(self.get_rotation(self.pos, self.ori, best[0])):
                    best = p, dist
            else:
                continue
        ang = self.get_rotation(self.pos, self.ori, best[0])
        result = None
        if 0 == ang:
            result = 0, 1
        if 0 < ang:
            result = -90, 0
        if ang < 0:
            result = 90, 0
        return result

    def update_map_from(self, pos):
        """
        Update map from given position so that every cell except the goal have
        value (1 + value of smallest neighbour values).
        """
        stack = [pos]
        while 0 < len(stack):
            pos = numpy.asarray(stack.pop(0))
            # do not update goal
            if all(x in self.goal for x in pos):
                continue
            tmp = []
            min_dist = sys.maxint
            for m in self.moves:
                neib = pos + m
                if self.is_valid(*neib) and self.map.is_connected(pos, neib):
                    tmp.append(neib)
                    if self.map[neib] < min_dist:
                        min_dist = self.map[neib]
            assert(min_dist != sys.maxint)
            if self.map[pos] < min_dist + 1:
                logger.info(pos)
                logger.info("before {}".format(self.map[pos]))
                self.map[pos] = min_dist + 1
                self.image.update_text(pos, min_dist + 1)
                logger.info("after {}".format(self.map[pos]))
                stack.extend(tmp)


