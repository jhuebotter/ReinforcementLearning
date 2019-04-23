import numpy as np
import time
import tkinter as tk

UNIT = 100   # pixels
SIZE = int(UNIT / 2.3)
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width
crack = [(1,1),(1,3),(2,3),(3,1),(3,2),(3,3)]
ship = (2,2)
goal = (0,3)
start = (3,0)


class World(tk.Tk, object):
    def __init__(self):
        super(World, self).__init__()
        self.title('World')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_world()


    def _build_world(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create cracks
        for c in crack:
            center = np.array([UNIT/2 + c[1] * UNIT, UNIT/2 + c[0] * UNIT])
            self.canvas.create_rectangle(
            center[0] - SIZE, center[1] - SIZE,
            center[0] + SIZE, center[1] + SIZE,
            fill='black')

        # create goal
        goal_center = np.array([UNIT/2 + goal[1] * UNIT, UNIT/2 + goal[0] * UNIT])
        self.canvas.create_oval(
            goal_center[0] - SIZE, goal_center[1] - SIZE,
            goal_center[0] + SIZE, goal_center[1] + SIZE,
            fill='green')

        # create ship
        ship_center = np.array([UNIT/2 + ship[1] * UNIT, UNIT/2 + ship[0] * UNIT])
        self.canvas.create_oval(
            ship_center[0] - SIZE, ship_center[1] - SIZE,
            ship_center[0] + SIZE, ship_center[1] + SIZE,
            fill='yellow')

        # create agent
        start_pos = np.array([UNIT/2 + start[1] * UNIT, UNIT/2 + start[0] * UNIT])
        self.rect = self.canvas.create_rectangle(
            start_pos[0] - SIZE, start_pos[1] - SIZE,
            start_pos[0] + SIZE, start_pos[1] + SIZE,
            fill='red')

        # pack all
        self.canvas.pack()


    def reset(self, start):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        start_pos = np.array([UNIT/2 + start[1] * UNIT, UNIT/2 + start[0] * UNIT])
        self.rect = self.canvas.create_rectangle(
            start_pos[0] - SIZE, start_pos[1] - SIZE,
            start_pos[0] + SIZE, start_pos[1] + SIZE,
            fill='red')


    def move(self, pos, new_pos):
        y = (new_pos[0] - pos[0]) * UNIT
        x = (new_pos[1] - pos[1]) * UNIT
        #print(x,y)
        self.canvas.move(self.rect, x, y)  # move agent


    def render(self):
        time.sleep(1.0)
        self.update()


def update(pos=start, new_pos=(2,0)):
    #s = env.reset(start)
    env.render()
    env.move(pos, new_pos)


if __name__ == '__main__':
    env = World()
    env.after(100, update)
    env.mainloop()