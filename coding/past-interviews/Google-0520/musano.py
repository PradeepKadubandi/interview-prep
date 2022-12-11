class Node:
    def __init__(obj, n=None):
        self.obj = obj
        self.next = n

dummy = Node('DUMMY')

def generateSequence(node):
    res = []
    while node:
        res.append(node.obj[0])
        node = node.next
    return res

def musano(A, B, M):
    c = dummy
    m = {}
    while (A,B) not in m:
        n = Node((A,B))
        m[(A,B)] = c.next = n
        c = c.next
        A, B = B, A*B % M
    hRepeat = m[(A,B)]
    c = dummy

    # Below code block does only single traversal over list
    # but there is code duplication
    # version1: refactoring it makes code worse - because notice how the order of adding elements to list is different
    # version2: The second loop is counter intuitive but modified to allow for refactoring
    leading = []
    while c.next != hRepeat:
        c = c.next
        leading.append(c.obj[0])
    repeat = []




    
    while c.next:
        c = c.next
        repeat.append(c.obj[0])
    return leading, repeat

    # Below code block does 2 passes on leading list but better
    # modular and readable code
    # while c.next != hRepeat:
    #     c = c.next
    # c.next = None
    # return generateSequence(dummy.next), generateSequence(hRepeat)