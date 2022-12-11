import heapq

class Heap:
    def __init__(self, data = [], is_max_heap = False, negate = lambda x: -1 * x):
        '''
        Object Oriented Heap for python!
        Use 'data' argument if there is an initial list that needs to be conveted to heap.
        Set 'is_max_heap' to True if max heap behavior is desired.
        When max heap behavior is on, negate parameter is used to change the element to enfore the right order.
        The default negate function assumes numbers as data type for heap. For other data types (for ex: string),
        an appropriate negate function must be provided.
        '''
        self.data = data
        self.is_max_heap = is_max_heap
        self.negate = negate
        if len(self.data) > 0:
            if self.is_max_heap:
                self.data = [self.negate(x) for x in self.data]
            heapq.heapify(self.data)

    def push(self, elem):
        elem = self.__negate_if_needed(elem)
        heapq.heappush(self.data, elem)

    def pop(self):
        elem = None
        if len(self.data) > 0:
            elem = heapq.heappop(self.data)
        return self.__negate_if_needed(elem)

    def peek(self):
        if len(self.data) == 0:
            return None
        return self.__negate_if_needed(self.data[0])

    def __negate_if_needed(self, elem):
        if elem and self.is_max_heap:
            return self.negate(elem)
        return elem

class Tester:
    def __init__(self, testHeap):
        self.testHeap = testHeap
    
    def test(self, operations):
        actual = []
        expected = []
        for op, elem in operations:
            if op == 'push':
                self.testHeap.push(elem)
            elif op == 'pop':
                expected.append(elem)
                actual.append(self.testHeap.pop())
            elif op == 'peak':
                expected.append(elem)
                actual.append(self.testHeap.peek())
        if len(expected) != len(actual):
            self.__report_failure(expected, actual)
            return
        for i, v in enumerate(expected):
            if actual[i] != v:
                self.__report_failure(expected, actual)
                return

    def __report_failure(self, expected, actual):
        print ('Test Failed : expected = {}, actual = {}', expected, actual)

print ('Min heap with numbers test:')
min_heap_numbers_test = Tester(Heap())
min_heap_numbers_test.test([
    ('push', 10), ('peek', 10),
    ('push', 20), ('peek', 10),
    ('push', 5), ('peek', 5),
    ('pop', 5), ('peek', 10),
    ('pop', 10), ('peek', 20),
    ('pop', 20), ('peek', None),
    ])

print ('Max heap with numbers test:')
max_heap_numbers_test = Tester(Heap(is_max_heap=True))
max_heap_numbers_test.test([
    ('push', 10), ('peek', 10),
    ('push', 20), ('peek', 20),
    ('push', 5), ('peek', 20),
    ('pop', 20), ('peek', 10),
    ('pop', 10), ('peek', 5),
    ('pop', 5), ('peek', None),
    ])

# This negate formulation for string is incorrect ...
# A better approach is to use a wrapper object over string that overrides element wide comparison and use that as data element for heap
# https://stackoverflow.com/questions/2501457/what-do-i-use-for-a-max-heap-implementation-in-python
print ('Max heap with string test:')
max_heap_strings_test = Tester(Heap(is_max_heap=True, negate= lambda x: x[::-1]))
max_heap_strings_test.test([
    ('push', 'abc'), ('peek', 'abc'),
    ('push', 'zebra'), ('peek', 'zebra'),
    ('push', 'middle'), ('peek', 'zebra'),
    ('pop', 'zebra'), ('peek', 'middle'),
    ('pop', 'middle'), ('peek', 'abc'),
    ('pop', 'abc'), ('peek', None),
    ])

class MaxHeapObj(object):
  def __init__(self, val): self.val = val
  def __lt__(self, other): return self.val > other.val
  def __eq__(self, other): return self.val == other.val
  def __str__(self): return str(self.val)

print ('Max heap with wrapper objects over string test:')
max_heap_strings_test = Tester(Heap())
max_heap_strings_test.test([
    ('push', MaxHeapObj('abc')), ('peek',  MaxHeapObj('abc')),
    ('push', MaxHeapObj('zebra')), ('peek',  MaxHeapObj('zebra')),
    ('push', MaxHeapObj('middle')), ('peek',  MaxHeapObj('zebra')),
    ('pop',  MaxHeapObj('zebra')), ('peek',  MaxHeapObj('middle')),
    ('pop',  MaxHeapObj('middle')), ('peek',  MaxHeapObj('abc')),
    ('pop',  MaxHeapObj('abc')), ('peek', None),
    ])

print ('Done')
