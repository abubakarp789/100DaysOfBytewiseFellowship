num = int(input("How many terms?: "))
x = 0
y = 1
print("Fibanocci Sequence: ")
print(x, y, end= ' ')
for i in range(num):
    c = x + y
    x = y
    y = c
    print(c, end = " ")