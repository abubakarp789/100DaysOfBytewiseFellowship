operator = input("Enter the operator(+, -, *, /): ")

if operator != "+" and operator != "-" and operator != "*" and operator != "/":
    print(f"{operator} invalid!!")
else:
    num1 = float(input("Enter the First number: "))
    num2 = float(input("Enter the second number: "))
    if operator == "+":
        result = num1 + num2
        print(round(result, 3))
    elif operator == "-":
        result = num1 - num2
        print(round(result, 3))
    elif operator == "*":
        result = num1 * num2
        print(round(result, 3))
    elif operator == "/":
        result = num1 / num2
        print(round(result, 3))
    