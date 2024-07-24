num1 = int(input("Enter the First number: "))
num2 = int(input("Enter the second number: "))
num3 = int(input("Enter the third number: "))

if num1 >= num2 and num1 >= num3:
    print(f"The Largest Number is: {num1}")
elif num2 >= num1 and num2 >= num3:
    print(f"The Largest Number is: {num2}")
elif num3 >= num1 and num3 >= num2:
    print(f"The Largest Number is: {num3}")