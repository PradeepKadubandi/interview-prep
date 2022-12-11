class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        self.cumMatrix = [[0 for j in range(len(matrix[i]))] for i in range(len(matrix))]
        if len(matrix) > 0 and len(matrix[0]) > 0:
            self.cumMatrix[0][0] = matrix[0][0]
            for i in range(1, len(matrix)):
                self.cumMatrix[i][0] = self.cumMatrix[i-1][0] + matrix[i][0]
            for j in range(1, len(matrix[0])):
                self.cumMatrix[0][j] = self.cumMatrix[0][j-1] + matrix[0][j]
            for i in range(1, len(matrix)):
                for j in range(1, len(matrix[i])):
                    self.cumMatrix[i][j] = matrix[i][j] + self.cumMatrix[i-1][j] + self.cumMatrix[i][j-1] - self.cumMatrix[i-1][j-1]
                
        

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        result = self.cumMatrix[row2][col2]
        if col1 > 0:
            result -= self.cumMatrix[row2][col1-1]
        if row1 > 0:
            result -= self.cumMatrix[row1-1][col2]
        if col1 > 0 and row1 > 0:
            result += self.cumMatrix[row1-1][col1-1]
        return result
        


# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)