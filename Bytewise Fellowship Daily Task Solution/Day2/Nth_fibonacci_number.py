number = int(input("Enter a number: "))

x = 0
y = 1
print("Fibanocci Sequence: ")
for i in range(number-1):
    c = x + y
    x = y
    y = c
print(f"The {number}th fibonacci number is {c}")