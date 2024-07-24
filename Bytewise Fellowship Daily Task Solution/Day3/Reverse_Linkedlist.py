class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def createLinkedList(values):
    head = None
    prev = None
    for val in values:
        node = ListNode(val)
        if head is None:
            head = node
        if prev is not None:
            prev.next = node
        prev = node
    return head

def reverseLinkedList(head):
    prev = None
    curr = head

    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node

    return prev

# Example usage
# Create the linked list: 1 -> 2 -> 3 -> 4 -> 5
values = [1, 2, 3, 4, 5]
head = createLinkedList(values)

# Reverse the linked list
reversed_head = reverseLinkedList(head)

# Print the reversed linked list
while reversed_head:
    print(reversed_head.val, end=" -> ")
    reversed_head = reversed_head.next
print("None")