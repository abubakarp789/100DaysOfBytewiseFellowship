# Get the first word from the user
word1 = input("Enter the first word: ")

# Get the second word from the user
word2 = input("Enter the second word: ")

# Convert both words to lowercase and remove any spaces
word1 = word1.lower().replace(" ", "")
word2 = word2.lower().replace(" ", "")

# Sort the characters in both words
sorted_word1 = sorted(word1)
sorted_word2 = sorted(word2)

# Check if the sorted words are equal
if sorted_word1 == sorted_word2:
    print(f"{word1} and {word2} are anagrams.")
else:
    print(f"{word1} and {word2} are not anagrams.")