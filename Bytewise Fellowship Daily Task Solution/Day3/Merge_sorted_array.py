def mergearray(arr1, arr2):
    i = 0
    j = 0
    arr = []
    
    # Merge the two arrays in sorted order
    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            arr.append(arr1[i])
            i += 1
        else:
            arr.append(arr2[j])
            j += 1
    
    # Append the remaining elements of arr1
    while i < len(arr1):
        arr.append(arr1[i])
        i += 1
    
    # Append the remaining elements of arr2
    while j < len(arr2):
        arr.append(arr2[j])
        j += 1
    
    return arr

print(mergearray([1, 3, 5], [2, 4, 6, 8, 10])) 

def mergethreesortedarray(arr1, arr2, arr3):
    i = mergearray(arr1, arr2)
    return mergearray(i, arr3)

print(mergethreesortedarray([1, 3, 5], [2, 4, 6, 8, 10], [7, 9, 11, 13, 15]))
