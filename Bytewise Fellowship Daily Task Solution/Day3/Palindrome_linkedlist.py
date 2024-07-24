def reverse(head):
    prev = None
    while head is not None:
        next_node = head.next
        head.next = prev
        prev = head
        head = next_node
    return prev

class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next

def Palindrome_Linkedlist(head):
    if head is None:
        return True
    slow = head
    fast = head
    while fast.next is not None and fast.next.next is not None:
        slow = slow.next
        fast = fast.next.next
    slow = slow.next
    slow = reverse(slow)
    fast = head
    while slow is not None:
        if slow.data != fast.data:
            print("This LinkedList is not a Palindrome")
        slow = slow.next
        fast = fast.next
    print("This LinkedList is a Palindrome")

Palindrome_Linkedlist(Node(1, Node(2, Node(2, Node(1)))))