import sys
import heapq
import argparse
import time

'''
Whats New:
  - ** Note that this file returns a different path for 2nd input
    when using BFS, however the path is valid and of same length as
    expected output **.
  - Used a different space representation for graph. Did not have a chance
    to do deep memory usage tests but the approach could potentially reduce
    the memory used to store the graph, idea is simple, given the maze may
    have lot of unavailable spots, avoid storing blocked spots in graph storage.
    Instead of 2-dimentional lists to store graph, use a list of dictionaries
    (each representing a row) and the dictionary keys correspond to only empty
    spaces and does not store indices for blocked cells. 
    - Arguable because we are storing the keys explicitly in dictionary whereas
      list indices are implicit from memory structure.
  - Corrected comments about time complexity of AStar because there is a priority
    queue involved whose inserts take an additional log N making the complexity
    N log N (N being the total input size). 
General comments:
  - Implemented standard Breadth First Search Graph Traversal to solve problem.
    Linear in time and space complexity w.r.t input size.
      - Both parent node tracking (for path retrieval) and visitor flag are
        implemented by modifying input graph in memory instead of using extra space.
  - Intuitively for a Maze, I wasn't sure how much AStar with manhattan distance heuristic
    will help. I tried it as I was curious.
      - Same space complexity as BFS, time complexity is N log N. 
        However search could be faster for some
        inputs (potentially when there is a relatively forward moving path). 
        However, for input3 which is considerably large,
        AStar performed worse w.r.t time. (Hence kept the default command line option as BFS)
  - Implemented a custom timeit measurement to measure just the algorithm
    part (excluding file IO) for specified number of iterations to report time.
      - This is the part that requires python3.
  - Usage:
      - python3 mazeSolver.py input1.txt --out output1.txt : Finds shortest path using BFS and writes to output1.txt
      - python3 mazeSolver.py input1.txt --astar : Finds shortest path using AStar and writes to output.txt
      - python3 mazeSolver.py input3.txt --timeit 10 : Processes the same file 10 times, only measuring time for core algorithm excluding IO.
'''
parser = argparse.ArgumentParser(description='Solve Maze')
parser.add_argument('inputFile', type=str, help='path to file containing maze')
parser.add_argument('--out', metavar='OutputFile', type=str, default='output.txt', help='path to output file, file on disk will be overwritten!')
parser.add_argument('--astar', action='store_true', help='Use AStar (by default BFS is used)')
parser.add_argument('--noout', action='store_true', help='Do not write any messages to console')
parser.add_argument('--timeit', metavar='Number', type=int, default=1)

args = parser.parse_args()

measure_mode = args.timeit > 1
if measure_mode:
    args.noout = True

def print_wrapper(message):
    if not args.noout:
        print (message)

class Maze:
    def __init__(self, source, target, graph, rows, cols):
        self.source = source
        self.target = target
        self.graph = graph
        self.rows = rows
        if self.rows == 0:
            raise 'Empty Graph'
        self.cols = cols
        self.directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        print_wrapper ('Maze size = {} rows and {} cols'.format(self.rows, self.cols))
        print_wrapper ('Source = {}, Target = {}'.format(self.source, self.target))

    def isSafeIndex(self, pos):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def getEdges(self, vertex):
        for dIndex, d in enumerate(self.directions):
            r = vertex[0] + d[0]
            c = vertex[1] + d[1]
            if self.isSafeIndex((r,c)) and self.graph[r].get(c, '#') in [' ', 'B']:
                yield dIndex, (r, c)

    def markPath(self, pathEnd):
        if pathEnd != None:
            L = 0
            print_wrapper ('Path Found')
            curr = pathEnd
            while curr != self.source:
                L += 1
                assert curr[1] in self.graph[curr[0]], 'markPath: Visting a non existing node in graph'
                dIndex = int(self.graph[curr[0]][curr[1]]) - 1
                direction = self.directions[dIndex]
                parent = (curr[0] - direction[0], curr[1] - direction[1])
                self.graph[curr[0]][curr[1]] = '.'
                curr = parent
            print ('Path length: {}'.format(L))
        else:
            print_wrapper ('Path not found')
            self.writeMaze(args.out, debug=True)

    def writeMaze(self, outputPath, debug=False):
        with open(outputPath, 'w') as f:
            for row in range(self.rows):
                for col in range(self.cols):
                    c = self.graph[row].get(col, '#')
                    if c in ['.', '#', 'A', 'B'] or debug:
                        f.write(c)
                    else:
                        f.write(' ')
                f.write('\n')

    def BFS(self):
        '''
        Standard Breadth First Seach Graph Traversal.
        Note that this implementation does not use any additional memory than necessary,
        For example, a separate visited flag array is not used to mark visited vertices.
        The code updates the input graph array to
        store indication for parent to later retrieve the path (and uses that as visited flag too).
        Of course, the algorithm uses a FIFO Q and the space complexity of
        algorithm is linear in terms of input size.
        '''
        Q = [self.source]
        pathEnd = None
        while len(Q) > 0:
            curr = Q.pop()
            for dIndex, edge in self.getEdges(curr):
                if edge == self.target:
                    pathEnd = curr
                    break
                # Mark the node as visited by setting parent (done by storing dIndex from parent to here as a character)
                assert edge[1] in self.graph[edge[0]], 'BFS: Visting a non existing node in graph'
                self.graph[edge[0]][edge[1]] = str(1 + dIndex)
                Q.append(edge)
        self.markPath(pathEnd)

    def AStar(self):
        '''
        A* with manhattan distance to goal as heuristic cost to reach goal.
        Uses Priority Queue with priority = (cost-from-source + heurisitic-cost-to-goal).
        The memory complexity is still linear in terms of input size though there's more overhead than BFS.

        Interesting Note about implementation:
          For general A* algorithm (where edge costs are variable), we need
          to run the algorithm till we pop 'goal' out of priority queue, otherwise,
          theoritically a sub-optimal path maybe returned.
          However, in our case, the edge costs are all considered 1, so for all
          neighbors around 'goal', the heurisitc cost is equal, so we will end up expanding
          the node with least cost from source among neighbors. (We can prove this by using
          contradiction - if we assume we explored a neighbor with higher cost from source 
          first, since the heuristic cost is same among all neighbors of goal, the total
          cost must also be higher for currently expanding neighbor but that's a contradiction
          because we expand nodes from a priority queue ordered based on total cost.)
          So in the implementation, I stopped when I see 'goal' in children (i.e., before adding
          to priority queue itself instead of waiting for it to be removed from PQ.) Though
          it doesn't change the time complexity, it stops search faster.
        '''
        def manhattan_heuristic(vertex, goal=self.target):
            return abs(goal[0]-vertex[0]) + abs(goal[1]-vertex[1])
        def total_cost(vertex, cost_from_source):
            return cost_from_source + manhattan_heuristic(vertex)

        # Map used as a quick way to implementing lowering of priorities in priority queue.
        # Idea is to use lazy deletion of old entries of the same cell from priority queue.
        # We add a new entry in priority queue for a given cell only when we see a lower
        # cost than previously existing one (done using map), we don't delete the previous
        # entry for this cell from PQ, but PQ implementation will make sure we visit the entry
        # with least cost first. Since we mark the cell as visited at this time, when
        # we see the same cell again later from priority queue with higher total cost,
        # we ignore this item (deletion on access).
        # (Some Reference : https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes)
        cost_map = dict() 
        visited = set()
        # Heap's items are of structure: (total_cost, cost_from_source, vertex)
        PQ = list() 

        # Theoritically first value should be non-zero but it doesn't matter since at this point it's one node in Heap.
        heapq.heappush(PQ, (0, 0, self.source)) 
        cost_map[self.source] = 0

        pathEnd = None
        while len(PQ) > 0:
            _, cost_from_source, vertex = heapq.heappop(PQ)
            if vertex not in visited:
                visited.add(vertex)
                for dIndex, edge in self.getEdges(vertex):
                    if edge == self.target:
                        pathEnd = vertex
                        break
                    new_total_cost = total_cost(edge, cost_from_source + 1)
                    if (edge not in cost_map) or (cost_map[edge] > new_total_cost):
                        cost_map[edge] = new_total_cost
                        assert edge[1] in self.graph[edge[0]], 'AStar: Visting a non existing node in graph'
                        self.graph[edge[0]][edge[1]] = str(1 + dIndex)
                        heapq.heappush(PQ, (new_total_cost, cost_from_source + 1, edge))
        self.markPath(pathEnd)

def readMaze(filePath):
    '''
    Instead of adjacency matrix which requires Rows * Cols number of bytes
    to store graph, uses an adjacency list to reduce the memory storage (arguable).
    (Implemented as a list of dictionaries, each dictionary representing
    characters in one row, the catch is '#' cells are not in dictionary
    saving space).
    '''
    graph = []
    source = target = None
    with open(filePath, 'r') as f:
        rows = 0
        for line in f:
            if len(line.strip()) == 0:
                break
            rows += 1
            cols = len(line)
            row = dict()
            for i,c in enumerate(line):
                if c == 'A':
                    source = (len(graph), i)
                if c == 'B':
                    target = (len(graph), i)
                if c != '#':
                    row[i] = c
            graph.append(row)
    return Maze(source, target, graph, rows, cols)

def main(args):
    total = 0.0
    for i in range(args.timeit):
        m = readMaze(args.inputFile)
        f = Maze.BFS
        if args.astar:
            f = Maze.AStar
            print_wrapper ('Using AStar')
        else:
            print_wrapper ('Using BFS')
        start = time.perf_counter()
        f(m)
        stop = time.perf_counter()
        total += stop - start
        if i == 0:
            m.writeMaze(args.out)
    if measure_mode:
        print ('Total Time = {}, Iterations = {}, Average per iteration = {}'
            .format(total, args.timeit, total/args.timeit))

main(args)