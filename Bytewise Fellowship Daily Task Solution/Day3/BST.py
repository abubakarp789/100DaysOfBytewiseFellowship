class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

def BST_insert(root, data):
    if root is None:
        return Node(data)
    if data < root.data:
        root.left = BST_insert(root.left, data)
    else:
        root.right = BST_insert(root.right, data)
    return root


def min_value(root):
    current = root
    while current.left is not None:
        current = current.left
    return current.data

def BST_delete(root, data):
    if root is None:
        return root
    if data < root.data:
        root.left = BST_delete(root.left, data)
    elif data > root.data:
        root.right = BST_delete(root.right, data)
    else:
        if root.left is None:
            return root.right
        elif root.right is None:
            return root.left
        root.data = min_value(root.right)
        root.right = BST_delete(root.right, root.data)
    return root

def BST_search(root, data):
    if root is None or root.data == data:
        return root
    if root.data < data:
        return BST_search(root.right, data)
    return BST_search(root.left, data)

def BST_print(root):
    if root is not None:
        BST_print(root.left)
        print(root.data, end = " ")
        BST_print(root.right)

root = None
root = BST_insert(root, 50)
root = BST_insert(root, 30)
root = BST_insert(root, 20)
root = BST_insert(root, 40)
root = BST_insert(root, 70)
root = BST_insert(root, 60)
root = BST_insert(root, 80)

# Print the elements of the BST in ascending order
print("Elements in the BST:")
BST_print(root)
print("")
# Delete node with value 20
print("After deleting 20:")
root = BST_delete(root, 20)
BST_print(root)
print("")
# Delete node with value 30
print("After deleting 30:")
root = BST_delete(root, 30)
BST_print(root)
print("")
# Delete node with value 50
print("After deleting 50:")
root = BST_delete(root, 50)
BST_print(root)
