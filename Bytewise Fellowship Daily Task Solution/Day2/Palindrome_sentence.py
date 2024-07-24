import string

# Get input from the user
user_input = input("Enter a sentence: ")

# Remove punctuation and convert to lowercase
cleaned_input = ''.join(c for c in user_input if c not in string.punctuation)
cleaned_input = cleaned_input.replace(" ", "").lower()

# Reverse the cleaned string
reversed_input = cleaned_input[::-1]

# Check if the reversed string is equal to the cleaned string
if reversed_input == cleaned_input:
    print(user_input + " is a palindrome.")
else:
    print(user_input + " is not a palindrome.")