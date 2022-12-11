# Linked List with integer nodes

class Node:
    def __init__(self, data, n):
        self.data = data
        self.n = n
        
def isOdd(a):
    return a % 2 == 1

def deleteOddIntegers(head):
    while head != None and isOdd(head.data):
        head = head.n

    if head == None:
        return None
    # if head.n == None:
    #     if isOdd(head.data):
    #         return None
    #     else:
    #         return head

    curr = head
    while (curr.n != None):
        if isOdd(curr.n.data):
            curr.n = curr.n.n
        else:
            curr = curr.n
    return head

def printList(head):
    curr = head
    while (curr != None):
        print ('{},'.format(curr.data))
        curr = curr.n
        
# head = Node(1, Node(5, Node(1, Node(3, Node(7, None)))))
head = Node(1, None)

result = deleteOddIntegers(head)
printList(result)
