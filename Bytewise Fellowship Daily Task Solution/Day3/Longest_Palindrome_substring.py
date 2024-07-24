def expand_around_center(s, left, right):
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    return right - left - 1

def longest_palindrome_substring(s):
    n = len(s)
    if n == 0:
        return ""
    start = 0
    end = 0
    for i in range(n):
        len1 = expand_around_center(s, i, i)
        len2 = expand_around_center(s, i, i + 1)
        max_len = max(len1, len2)
        if max_len > end - start:
            start = i - (max_len - 1) // 2
            end = i + max_len // 2
    return s[start:end + 1]

print(longest_palindrome_substring("babad"))
print(longest_palindrome_substring("cbbd"))