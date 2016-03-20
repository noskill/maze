#!/usr/bin/python
from maze import Maze
import pygame
from pygame.locals import *
import sys
import numpy


def main():
    # Initialise screen
    pygame.init()
    # Create a maze based on input argument on command line.
    testmaze = Maze( str(sys.argv[1]) )

    size = 50 * testmaze.dim + 100
    screen = pygame.display.set_mode((size, size))
    pygame.display.set_caption('Basic Pygame program')

    # Fill background
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((250, 250, 250))
    foreground = pygame.Surface(screen.get_size())

    centr_x = background.get_rect().centerx
    centr_y = background.get_rect().centery

    # Blit everything to the screen
    screen.blit(background, (0, 0))
    pygame.display.flip()
    sq_size = 50

    # maze centered on (centr_x, centr_y), squares are 40 pixels in length.
    origin = numpy.asarray([centr_x, centr_y]) + [(testmaze.dim * sq_size / -2) for x in range(2)]
    max_coord = origin + testmaze.dim * sq_size

    arrow = pygame.transform.scale(pygame.image.load('./arrow up.png'), [sq_size - sq_size/4 for _ in range(2)])
    box = arrow.get_rect()
    pos = origin + [box.width/4, box.height]
    print pos

    def flip_y(array):
        return array * [1, -1] +  [0, max_coord[1] + origin[1]]


    # iterate through squares one by one to decide where to draw walls
    for x in range(testmaze.dim):
        for y in range(testmaze.dim):
          if not testmaze.is_permissible([x,y], 'up'):
              start_pos =  origin + [sq_size * x, sq_size * (y+1)]
              end_pos = start_pos + [sq_size, 0]
              pygame.draw.line(background, (0,0,0), flip_y(start_pos), flip_y(end_pos))

          if not testmaze.is_permissible([x,y], 'right'):
              start_pos =  origin + [sq_size * (x+1), sq_size * y]
              end_pos = start_pos + [0, sq_size]
              pygame.draw.line(background, (0,0,0), flip_y(start_pos), flip_y(end_pos))

          # only check bottom wall if on lowest row
          if y == 0 and not testmaze.is_permissible([x,y], 'down'):
              start_pos =  origin + [sq_size * x, 0]
              end_pos = start_pos + [sq_size, 0]
              pygame.draw.line(background, (0,0,0), flip_y(start_pos), flip_y(end_pos))

          # only check left wall if on leftmost column
          if x == 0 and not testmaze.is_permissible([x,y], 'left'):
              start_pos =  origin + [0, sq_size * y]
              end_pos =  start_pos + [0, sq_size]
              pygame.draw.line(background, (0,0,0), flip_y(start_pos), flip_y(end_pos))
    newx = 15


    # Event loop
    while 1:
        for event in pygame.event.get():
            if event.type == QUIT:
                return
        screen.blit(background, (0, 0))

        #screen.blit(arrow, flip_y(pos))
        pygame.display.update()



if __name__ == '__main__':
    main()
