def merge_intervals(intervals):
    # Sort the intervals based on the start time of each interval
    intervals.sort(key=lambda x: x[0])
    merged = []  # List to store the merged intervals
    for interval in intervals:
        # If the merged list is empty or the current interval does not overlap with the previous interval
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)  # Add the current interval to the merged list
        else:
            # If the current interval overlaps with the previous interval, merge them by updating the end time
            merged[-1] = [merged[-1][0], max(merged[-1][1], interval[1])]
    return merged

intervals = [(1, 3), (2, 6), (8, 10), (15, 18)]
merged_intervals = merge_intervals(intervals)
print(merged_intervals)