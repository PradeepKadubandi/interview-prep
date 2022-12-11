import numpy as np

class Grid:
    def __init__(self, data):
        self.dim1 = len(data)
        self.dim2 = len(data[0]) if self.dim1 > 0 else 0
        self.data = data
    
    def isSafeIndex(self, i, j):
        return 0 <= i < self.dim1 and 0 <= j < self.dim2
    
    def children(self, i, j):
        for dh,dv in [(0, -1), (-1, 0), (0, 1), (1, 0)]:
            ch, cv = i+dh, j+dv
            if self.isSafeIndex(ch, cv):
                yield ch, cv
 
    def isVisited(self, i, j):
        return self.data[i][j] == '0'
        
    def setVisited(self, i, j):
        self.data[i][j] = '0'
        
    def dfs(self, i, j):
        self.setVisited(i, j)
        for ch, cv in self.children(i, j):
            if not self.isVisited(ch, cv):
                self.dfs(ch, cv)

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        islands = 0
        g = Grid(grid)
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j] == '1':
                    islands += 1
                    g.dfs(i, j)
        return islands