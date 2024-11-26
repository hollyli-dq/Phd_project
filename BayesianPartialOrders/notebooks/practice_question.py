#- Given an array of N elements, remove K elements to minimize the amplitude(A_max - A_min) of the remaining array. [DONE]

def min_amplitude(arr, K):
    N = len(arr)
    if K >= N:
        return 0  # Edge case where we remove too many elements
    
    arr.sort()
    
    # Size of the resulting array will be N - K
    min_amp = float('inf')
    for i in range(N - K):
        current_amp = arr[i + K] - arr[i]
    if current_amp < min_amp:
        min_amp = current_amp
    
    return min_amp
