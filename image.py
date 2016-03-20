import time
import numpy
import pygame
from math import pi


class Image(object):
    def __init__(self, dim, screen):
        """
        screen: pygame screen
        dim: int
        Height or width of the maze, assuming maze is a square
        """
        self.dim = dim
        self.screen = screen
        pygame.display.set_caption('Basic Pygame program')
        self.RLeft = numpy.asarray([[0, -1],[1, 0]])
        self.text_cache = {}
        self.text_map = {}

        # Fill background
        background = pygame.Surface(self.screen.get_size())
        self.background = background.convert()
        self.background.fill((250, 250, 250))

        centr_x = background.get_rect().centerx
        centr_y = background.get_rect().centery

        # Display some text
        self.font = pygame.font.Font(None, 24)
        text = self.font.render("Hello There", 1, (10, 10, 10))
        textpos = text.get_rect()
        textpos.centerx = centr_x
        textpos.centery = centr_y

        # Blit everything to the screen
        self.screen.blit(background, (0, 0))
        pygame.display.flip()
        self.sq_size = 50
        print textpos.centerx
        print textpos.centery

        # maze centered on (centr_x, centr_y), squares are 40 pixels in length.
        self.origin = numpy.asarray([centr_x, centr_y], dtype='float64') + [(self.dim * self.sq_size / -2.0) for _ in range(2)]

        self.max_coord = self.origin + self.dim * self.sq_size
        self.draw_box()
        self.pos = None
        self.arrow = None
        self.new_pos = None
        self.new_ang = 0
        self.reset_arrow()

    def reset_arrow(self):
        """
        Reset arrow to initial position and orientation
        """
        self.arrow = pygame.transform.scale(pygame.image.load('./arrow up.png'), [self.sq_size - self.sq_size/4 for _ in range(2)])
        self.pos = self.origin + self.sq_size / 2.0
        self.new_pos = self.pos
        self.new_ang = 0

    def draw_box(self):
        """
        Draw perimeter of the maze
        """
        origin = self.origin
        start_pos =  origin
        end_pos = start_pos + [self.sq_size * self.dim, 0]
        pygame.draw.line(self.background, (0,0,0), self._flip_y(start_pos), self._flip_y(end_pos))

        end_pos = start_pos + [0, self.sq_size * self.dim]
        pygame.draw.line(self.background, (0,0,0), self._flip_y(start_pos), self._flip_y(end_pos))
        start_pos = origin + [0, self.sq_size * self.dim]
        end_pos = start_pos + [self.sq_size * self.dim, 0]

        pygame.draw.line(self.background, (0,0,0), self._flip_y(start_pos), self._flip_y(end_pos))

        start_pos = origin + [self.sq_size * self.dim, 0]
        pygame.draw.line(self.background, (0,0,0), self._flip_y(start_pos), self._flip_y(end_pos))

    def _flip_y(self, array):
        return array * [1, -1] +  [0, self.max_coord[1] + self.origin[1]]

    def draw_line_between(self, a_from, a_to):
        """
        draw wall between two cells
        """
        origin = self.origin + self.sq_size / 2
        t_from = a_from * self.sq_size + origin
        t_to = a_to * self.sq_size + origin
        center = (t_from + t_to) / 2
        start_pos = numpy.dot(self.RLeft,  t_from - center) * -1 + center
        end_pos = numpy.dot(self.RLeft,  t_from - center) * 1 + center
        pygame.draw.line(self.background, (0,0,0), self._flip_y(start_pos), self._flip_y(end_pos))

    def update_text(self, a_pos, a_value):
        value = str(a_value)
        if self.text_cache.get(value, None) == None:
            self.text_cache[value] = self.font.render(value, 1, (10, 10, 10))
        pos = self.origin + self.sq_size * numpy.asarray(a_pos) + self.sq_size / 2
        self.text_map[tuple(self._flip_y(pos))] = self.text_cache[value]

    def update(self, delta):
        """
        update postion of the mouse
        draw everything on the screen
        -------
        delta: float
        milliseconds since last frame
        -------
        Returns: None
        """
        self.screen.blit(self.background, (0, 0))
        for (p, t) in self.text_map.items():
            self.screen.blit(t, p)
        threshold = 0.1
        if threshold < abs(self.new_ang):
            # 90 degrees in one second
            #ang_delta = 90 * delta / 1000 * self.new_ang / abs(self.new_ang)
            ang_delta = self.new_ang
            if abs(self.new_ang) - abs(ang_delta) < 1:
                ang_delta = self.new_ang
            self.arrow = pygame.transform.rotate(self.arrow, ang_delta)
            self.new_ang -= ang_delta


        dist = numpy.linalg.norm(self.pos - self.new_pos)

        if threshold < dist:
            vec = self.new_pos - self.pos
            # sq_size in one second
            # to do 1 sec
            self.pos += vec / numpy.linalg.norm(vec) * float(delta) / 500 * self.sq_size
            if numpy.linalg.norm(self.pos - self.new_pos) < 2:
                self.pos = self.new_pos
        self.screen.blit(self.arrow, self._flip_y(self.pos))

    def move(self, pos):
        """
        Move mouse to new position
        """
        self.new_pos = self.origin + self.sq_size / 2.0 + (self.sq_size * pos)
        threshold = 0.1
        print self.new_pos
        while threshold < numpy.linalg.norm(self.pos - self.new_pos):
            time.sleep(0.1)

    def rotate(self, ang):
        """
        Rotate
        """
        self.new_ang = ang * 180 / pi
        threshold = 0.01
        while threshold < abs(self.new_ang):
            time.sleep(0.1)


class DummyImage(object):
    def __init__(self, *args):
        pass

    def draw_box(self):
        pass

    def move(self, pos):
        pass

    def rotate(self, ang):
        pass

    def update(self, delta):
        pass

    def reset_arrow(self):
        pass

    def update_text(self, a_pos, a_value):
        pass

    def draw_line_between(self, a_from, a_to):
        pass
