from typing import List
from collections import deque, namedtuple

# The problem was mine sweeper. Need to implement what happens
# in the backend of game implementation when user
# clicks on a cell.

class Grid:
    def __init__(self, rows, cols, mine_locations: List[List[int]]):
        self.rows = rows
        self.cols = cols
        self.grid = [[-1 for _ in range(cols)] for _ in range(rows)]
        for location in mine_locations:
            self.grid[location[0]][location[1]] = -9

    def _neighbors(self, parent):
        assert self._get_element(parent) != -9
        children = []
        x, y = parent
        mine_count = 0
        for dx, dy in [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]:
            cx, cy = x + dx, y + dy
            if 0 <= cx < self.rows and 0 <= cy < self.cols:
                child = (cx, cy)
                c_el = self._get_element(child)
                if c_el == -9:
                    mine_count += 1
                elif c_el == -1:
                    children.append(child)
        return children, mine_count

    def _get_element(self, pos):
        return self.grid[pos[0]][pos[1]]

    def _set_element(self, pos, val):
        self.grid[pos[0]][pos[1]] = val

    def expand_from_pos(self, src):
        el = self._get_element(src)
        if el == -9:
            raise Exception("Mine Clicked!")
        if el != -1:
            return

        children, mine_count = self._neighbors(src)
        self._set_element(src, mine_count)
        if mine_count == 0:
            for n in children:
                self.expand_from_pos(n)


    def print_grid(self):
        for row in range(self.rows):
            print (" , ".join(map(str, self.grid[row])))

class TestGrid:
    def print_test(self):
        test_obj = Grid(8, 6, mine_locations = [[1, 1], [0, 0], [5, 3], [4, 2], [4, 0], [4, 1], [6, 3], [7, 3]])
        test_obj.print_grid()
        print ("-------------------------")
        test_obj.expand_from_pos([5, 1])
        test_obj.print_grid()
        print ("-------------------------")
        test_obj.expand_from_pos([6, 1])
        test_obj.print_grid()
        print ("-------------------------")
        test_obj.expand_from_pos([0, 5])
        test_obj.print_grid()
    
t = TestGrid()
t.print_test()