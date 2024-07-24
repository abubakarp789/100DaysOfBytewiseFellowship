def min_edit_distance(str1, str2):
    # Base case: if either string is empty, return the length of the other string
    if len(str1) == 0:
        return len(str2)
    if len(str2) == 0:
        return len(str1)

    # If the last characters of the strings are the same, recursively solve for the remaining substrings
    if str1[-1] == str2[-1]:
        return min_edit_distance(str1[:-1], str2[:-1])

    # If the last characters are different, consider all possible operations (insert, delete, replace)
    insert_cost = 1 + min_edit_distance(str1, str2[:-1])
    delete_cost = 1 + min_edit_distance(str1[:-1], str2)
    replace_cost = 1 + min_edit_distance(str1[:-1], str2[:-1])

    # Return the minimum cost among the three operations
    return min(insert_cost, delete_cost, replace_cost)

# Test the function
str1 = "kitten"
str2 = "sitting"
print(min_edit_distance(str1, str2))
