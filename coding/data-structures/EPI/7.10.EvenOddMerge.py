import copy

class Node:
    def __init__(self, data, n=None):
        self.data = data
        self.next = n

def EvenOddMerge(head):
    if not head:
        return head
    evenHead = head.next
    curr = head
    i = 1
    while curr.next:
        i += 1
        bkp = curr.next
        curr.next = bkp.next
        prev = curr
        curr = bkp
    last = prev if i % 2 == 0 else curr
    last.next = evenHead
    return head

def stringifyList(h):
    head = h
    s = ''
    while head:
        s += str(head.data)
        head = head.next
        if head:
            s += ' --> '
    return s

def AreEqual(head1, head2):
    h1 = head1
    h2 = head2
    while h1 and h2 and h1.data == h2.data:
        h1 = h1.next
        h2 = h2.next
    return h1 == h2

tests = {
    1: (None, None),
    2: (Node(1), Node(1)),
    3: (Node(1, Node(2)), Node(1, Node(2))),
    4: (Node(1, Node(2, Node(3))), Node(1, Node(3, Node(2)))),
    5: (Node(1, Node(2, Node(3, Node(4)))), Node(1, Node(3, Node(2,  Node(4))))),
    6: (Node(1, Node(2, Node(3, Node(4, Node(5))))), Node(1, Node(3, Node(5,  Node(2, Node(4)))))),
}

for i,v in tests.items():
    t, exp = v
    act = EvenOddMerge(copy.deepcopy(t))
    success = AreEqual(exp, act)
    if not success:
        print ('Failed Test {}: Input = {}, Expected = {}, Actual = {}, AreEqual = {}'.format(
            i, stringifyList(t), stringifyList(exp), stringifyList(act), success
        ))

print ('Done')
