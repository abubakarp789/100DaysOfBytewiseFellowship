x = input("Enter a String: ")
y = ''

print(f"The origional String is: {x}")

for i in range(len(x)-1, -1, -1):
    y += x[i]

print(f"The Reverse String is: {y}")