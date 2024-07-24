user_input = input("Enter a string: ").lower()
reversed_input = user_input[::-1]
if reversed_input == user_input:
    print(user_input + " is a palindrome")
else:
    print(user_input + " is not a palindrome")
