'''
A 2d grid map of m rows and n columns is initially filled with water. We may perform an addLand operation which turns the water at position (row, col) into a land. Given a list of positions to operate, count the number of islands after each addLand operation. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.
Example:
Given m = 3, n = 3, positions = [[0,0], [0,1], [1,2], [2,1],[1,0],[0,2],[1,1]]. Initially, the 2d grid grid is filled with water. (Assume 0 represents water and 1 represents land).
0 0 0 
0 0 0
0 0 0
Operation #1: addLand(0, 0) turns the water at grid[0][0] into a land.
1 0 0
0 0 0   Number of islands = 1
0 0 0
Operation #2: addLand(0, 1) turns the water at grid[0][1] into a land.
1 1 0
0 0 0   Number of islands = 1
0 0 0
Operation #3: addLand(1, 2) turns the water at grid[1][2] into a land.
1 1 0
0 0 1   Number of islands = 2
0 0 0
Operation #4: addLand(2, 1) turns the water at grid[2][1] into a land.
1 1 0
0 0 1   Number of islands = 3
0 1 0
Operation #5: addLand(1, 0) turns the water at grid[1][0] into a land.
1 1 0
1 0 1   Number of islands = 3
0 1 0
Operation #6: addLand(0, 2) turn the water at grid[0][2] into a land
1 1 1
1 0 1   Number of islands = 2
0 1 0
Operation #7: addLand(1, 1) turn the water at grid[1][1] into a land
1 1 1
1 1 1   Number of islands = 1
0 1 0
We return the result as an array: [1, 1, 2, 3, 3, 2, 1]

'''
from collections import defaultdict
from typing import List, Tuple

# Doesn't yet work. Need to make it work!!!
class Node:
    def __init__(self, parent = None):
        self.parent = parent
        
class UnionAndFind:
    def __init__(self, rows, cols):
        self.parent_cells = [[(r, c) for c in range(cols)] for r in range(rows)]
        self.rank = [[0 for c in range(cols)] for r in range(rows)]
        self.rows = rows
        self.cols = cols
        self.count = 0
        
    def find(self, cell):
        while (self.parent_cells[cell[0]][cell[1]] != cell):
            return self.find(self.parent_cells[cell[0]][cell[1]])
        return cell
            
    def union(self, cell1, cell2):
        p1 = self.find(cell1)
        p2 = self.find(cell2)
        r1 = self.rank[p1]
        r2 = self.rank[p2]
        if r1 < r2:
            self.parent_cells[p1] = p2
        
        self.parent_cells[p2] = p1
        if r1 == r2:
            self.rank[p1] += 1
        self.count -= 1
    
    def add_island(self):
        self.count += 1
            
    def get_island_count(self):
        return self.count
    
def count_islands(rows, cols, positions: List[Tuple[int]]) -> List[int]:
    if rows == 0:
        return [0] * len(positions)
    
    grid = [[0 for c in range(cols)] for r in range(rows)]
    uf = UnionAndFind(rows, cols)
    
    result = list()
    for pos in positions:
        grid[pos[0]][pos[1]] == "1"
        uf.add_island()
        for d in [[-1, 0], [1, 0], [0, 1], [0, -1]]:
            nr, nc = (pos[0]+d[0], pos[1]+d[1])
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == "1":
                uf.union(pos, (nr, nc))
        result.append(uf.get_island_count())
        
    return result


print (count_islands(3, 3, [[0,0], [0,1], [1,2], [2,1],[1,0],[0,2],[1,1]]))

        