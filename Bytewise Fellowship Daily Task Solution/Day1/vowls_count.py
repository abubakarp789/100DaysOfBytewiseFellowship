x = input("Enter a String: ").lower()
count = 0
for i in x:
    match i:
        case 'a':
            count += 1
        case 'e':
            count += 1
        case 'i':
            count += 1
        case 'o':
            count += 1
        case 'u':
            count += 1

print(f"The number of vowls is: {count}")
        