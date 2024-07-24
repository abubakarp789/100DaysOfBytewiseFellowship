# Get input from the user
user_input = input("Enter a string: ")

# Convert the input to lowercase
lowercase_input = user_input.lower()

# Reverse the lowercase string
reversed_input = lowercase_input[::-1]

# Check if the reversed string is equal to the original string
if reversed_input == lowercase_input:
    print(user_input + " is a palindrome")
else:
    print(user_input + " is not a palindrome")