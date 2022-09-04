from bisect import bisect
import collections
import itertools
import random
import math
# array boot camp
def even_odd(A):
    next_even,next_odd = 0, len(A)-1
    while next_even<next_odd:
        if A[next_even]%2 == 0:
            next_even+=1
        else:
            A[next_even], A[next_odd] = A[next_odd], A[next_even]
            next_odd -= 1

# the dutch national flag problem
def dutch_flag_partition_1(pivot_index, A):
    pivot = A[pivot_index]
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            if A[j]<pivot:
                A[i], A[j]= A[j], A[i]
                break
    for i in reversed(range(len(A))):
        if A[i]<pivot:
            break
        for j in reversed(range(i)):
            if A[j]>pivot:
                A[i], A[j] = A[j], A[i]

def dutch_flag_partition_2(pivot_index, A):
    pivot = A[pivot_index]
    smaller = 0
    for i in range(len(A)):
        if A[i]<pivot:
            A[i], A[smaller]= A[smaller], A[i]
            smaller += 1
    larger = len(A)- 1
    for i in reversed(range(len(A))):
        if A[i]<pivot:
            break
        elif A[i]>pivot:
            A[i], A[larger] = A[larger], A[i]
            larger -= 1

def dutch_flag_partition_3(pivot_index, A):
    pivot = A[pivot_index]
    smaller ,equal, larger = 0, 0, len(A)
    while equal < larger:
        if A[equal]<pivot:
            A[smaller], A[equal]= A[equal], A[smaller]
            smaller, equal = smaller+1, equal+1
        elif A[equal] == pivot:
            equal += 1
        else:
            larger -= 1
            A[equal], A[larger] = A[larger], A[equal]

# increment an arbitrary-precision integer
def plus_one(A):
    A[-1] += 1
    for i in reversed(range(1, len(A))):
        if A[i] != 10:
            A[i] = 0
            A[i-1] += 1
            if A[0] == 10:
                A[0] = 1
                A.append(0)
    return A

# multiply two arbitrart-precision integers
def multiply(num1, num2):
    sign = -1 if ((num1[0]<0)^(num2[0]<0)) else 1
    num1[0], num2[0] = abs(num1[0]), abs(num2[0])
    result = [0] * (len(num1)+ len(num2))
    for i in reversed(range(len(num1))):
        for j in reversed(range(len(num2))):
            result[i+j+1] += num1[i]*num2[j]
            result[i+j]+= result[i+j+1]//10
            result[i+j+1]%= 10
    result = result[next((i for i, x in enumerate(result) if x!= 0), len(result)):] or [0]
    return [sign*result[0]+ result[1:]]

# advancing through an array
def can_read_end(A):
    furthest_reach_so_far , last_index = 0, len(A)-1
    i = 0
    while i<= furthest_reach_so_far and furthest_reach_so_far<last_index:
        furthest_reach_so_far = max(furthest_reach_so_far, A[i]+ i)
        i += 1
    return furthest_reach_so_far >= last_index

# delete duplicates from a sorted array
def delete_duplicates(A):
    if not A:
        return 0
    write_index = 1
    for i in range(1, len(A)):
        if A[write_index - 1]!= A[i]:
            A[write_index] = A[i]
            write_index += 1
    return write_index

# buy and sell a stock once
def buy_and_sell_stock_once(prices):
    min_price_so_far ,  max_profit = float('inf'), 0.0
    for price in prices:
        max_profit_sell_today = price - min_price_so_far
        max_profit = max(max_profit, max_profit_sell_today)
        min_price_so_far = min(min_price_so_far, price)
    return max_profit

# buy and sell a stock twice
def buy_and_sell_stock_twice(prices):
    max_total_profit, min_price_so_far = 0.0, float('inf')
    first_buy_sell_profits = [0]*len(prices)
    for i, price in enumerate(prices):
        min_price_so_far = min(min_price_so_far, price)
        max_total_profit = max(max_total_profit, price - min_price_so_far)
        first_buy_sell_profits[i] = max_total_profit
    max_price_so_far = float('-inf')
    for i, price in reversed(list(enumerate(prices[1:], 1))):
        max_price_so_far = max(max_price_so_far, price)
        max_total_profit = max(max_total_profit, max_price_so_far-price+first_buy_sell_profits[i-1])
    return max_total_profit

# enumerate all primes to n
def generate_primes_1(n):
    primes = []
    is_primes = [False, False]+[True]*(n-1)
    for p in range(2, n+1):
        if is_primes[p]:
            primes.append(p)
            for i in range(p, n+1, p):
                is_primes[i]= False
    return primes

def generate_primes_2(n):
    if n<2:
        return []
    size = (n-3)//2+1
    primes = [2]
    is_prime = [True]*size
    for i in range(size):
        if is_prime[i]:
            p = i*2+3
            primes.append(p)
            for j in range(2*i**2+6*i+3, size, p):
                is_prime[j] = False
    return primes

# permute the elements of an array
def apply_permutation_1(perm, A):
    for i in range(len(A)):
        next = i
        while perm[next]>=0:
            A[i], A[perm[next]] = A[perm[next]], A[i]
            temp = perm[next]
            perm[next] -= len(perm)
            next = temp
    perm[:] = [a+len(perm) for a in perm]

def apply_permutation_2(perm, A):
    def cyclic_permutation(start, perm, A):
        i, temp = start, A[start]
        while True:
            next_i = perm[i]
            next_temp = A[next_i]
            A[next_i] = temp
            i, temp = next_i, next_temp
            if i == start:
                break
    for i in range(len(A)):
        j = perm[i]
        while j!= i:
            if j<i:
                break
            j = perm[j]
        else:
            cyclic_permutation(i, perm, A)

# compute the next permutation
def next_permutation(perm):
    inversion_point = len(perm) - 2
    while (inversion_point>=0 and perm[inversion_point]>= perm[inversion_point+1]):
        inversion_point -= 1
    if inversion_point == -1:
        return []
    for i in reversed(range(inversion_point+1, len(perm))):
        if perm[i]>perm[inversion_point]:
            perm[inversion_point], perm[i] = perm[i], perm[inversion_point]
            break
    perm[inversion_point+1:] = reversed(perm[inversion_point+1:])
    return perm

# sample offline data
def randam_sampling(k, A):
    for i in range(k):
        r = random.randint(i, len(A)-1)
        A[i], A[r] = A[r], A[i]

# sample online data
def online_random_sample(it, k):
    sampling_result = list(itertools, itertools.islice(it, k))
    num_seen_so_far = k
    for x in it:
        num_seen_so_far += 1
        idx_to_replace = random.randrange(num_seen_so_far)
        if idx_to_replace < k:
            sampling_result[idx_to_replace] = x
    return sampling_result

# compute a random permutation
def compute_random_permutation(n):
    permutation = list(range(n))
    randam_sampling(n, permutation)
    return permutation

# compute a random subset
def random_subset(n, k):
    changed_elements = {}
    for i in range(k):
        rand_idx = random.randrange(i, n)
        rand_idx_mapped = changed_elements.get(rand_idx, rand_idx)
        i_mapped = changed_elements.get(i, i)
        changed_elements[rand_idx] = i_mapped
        changed_elements[i] = rand_idx_mapped
    return [changed_elements[i] for i in range(k)]

# generate nonuniform random numbers
def nonuniform_random_number_generation(values, probabilities):
    prefix_sum_of_probabilites = list(itertools.accumulate(probabilities))
    interval_idx = bisect.bisect(prefix_sum_of_probabilites, random.random())
    return values[interval_idx]

# the sudoku checker problem
def is_valid_sudoku(partial_assignment):
    def has_duplicate(block):
        block = list(filter(lambda x: x!= 0, block))
        return len(block)!= len(set(block))
    n = len(partial_assignment)
    if any(has_duplicate([partial_assignment[i][j]for j in range(n)])or has_duplicate([partial_assignment[j][i]for j in range(n)]) for i in range(n)):
        return False
    region_size = int(math.sqrt(n))
    return all(not has_duplicate([partial_assignment[a][b] for a in range(region_size*I, region_size*(I+1))
           for b in range(region_size*J, region_size*(J+1))]) for I in range(region_size) for J in range(region_size))

# rotate a 2D array
def rotate_matrix(square_matrix):
    matrix_size = len(square_matrix) - 1
    for i in range(len(square_matrix)//2):
        for j in range(i, matrix_size - 1):
            (square_matrix[i][j], square_matrix[~j][i], square_matrix[~i][~j], square_matrix[j][~i]) = (square_matrix[~j][i],
            square_matrix[~i][~j],square_matrix[j][~i], square_matrix[i][j])

class RotatedMatrix:
    def __init__(self, square_matrix):
        self._square_matrix = square_matrix
        
    def read_entry(self, i, j):
        return self._square_matrix[~j][i]

    def write_entry(self, i, j, v):
        self._square_matrix[~j][i] = v

# compute rows in pascal's triangle
def generate_pascal_triangle(n):
    result = [[1]*(i+1)for i in range(n)]
    for i in range(n):
        for j in range(1, i):
            result[i][j] = result[i - 1][j-1]+result[i-1][j]
    return result
    








    



