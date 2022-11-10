# Simple DES implementation
# @sutterseba
# Educational purpose only
import numpy as np


IP_TABLE = np.array([
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6,
    64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17,  9, 1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7
])

FP_TABLE = np.array([
    40, 8, 48, 16, 56, 24, 64, 32,
    39, 7, 47, 15, 55, 23, 63, 31,
    38, 6, 46, 14, 54, 22, 62, 30,
    37, 5, 45, 13, 53, 21, 61, 29,
    36, 4, 44, 12, 52, 20, 60, 28,
    35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26,
    33, 1, 41,  9, 49, 17, 57, 25
])

PC1_TABLE = np.array([
    57, 49, 41, 33, 25, 17, 9, 1,
    58, 50, 42, 34, 26, 18, 10, 2,
    59, 51, 43, 35, 27, 19, 11, 3,
    60, 52, 44, 36, 63, 55, 47, 39,
    31, 23, 15, 7, 62, 54, 46, 38,
    30, 22, 14, 6, 61, 53, 45, 37,
    29, 21, 13, 5, 28, 20, 12, 4
])

PC2_TABLE = np.array([
    14, 17, 11, 24, 1, 5, 3, 28,
    15, 6, 21, 10, 23, 19, 12, 4, 
    26, 8, 16, 7, 27, 20, 13, 2, 
    41, 52, 31, 37, 47, 55, 30, 40,
    51, 45, 33, 48, 44, 49, 39, 56,
    34, 53, 46, 42, 50, 36, 29, 32
])

E_TABLE = np.array([
    32, 1, 2, 3, 4, 5,
    4, 5, 6, 7, 8, 9,
    8, 9, 10, 11, 12, 13,
    12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21,
    20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29,
    28, 29, 30, 31, 32, 1
])

P_TABLE = np.array([
    16, 7, 20, 21, 29, 12, 28, 17,
    1, 15, 23, 26, 5, 18, 31, 10,
    2, 8, 24, 14, 32, 27, 3, 9,
    19, 13, 30, 6, 22, 11, 4, 25
])

# 8 substitution boxes, each 4 rows and 16 columns
S_BOX = np.empty((8, 4, 16), dtype=int)

S_BOX[0] = np.array([
    [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
    [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
    [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
    [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]
])

S_BOX[1] = np.array([
    [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
    [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
    [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
    [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]
])

S_BOX[2] = np.array([
    [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
    [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
    [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
    [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]
])

S_BOX[3] = np.array([
    [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
    [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
    [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
    [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]
])

S_BOX[4] = np.array([
    [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
    [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
    [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
    [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]
])

S_BOX[5] = np.array([
    [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
    [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
    [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
    [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]
])

S_BOX[6] = np.array([
    [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
    [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
    [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
    [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]
])

S_BOX[7] = np.array([
    [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
    [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
    [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
    [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]
])


def new_key (custom_seed = None):
    """returns a random key"""
    if custom_seed:
        np.random.seed(custom_seed)

    return np.random.randint(2, size=64)


def derive_keys (initial_key):
    """Schedules 16 round keys (48 bit) from initial key (64 bit)"""

    if initial_key.size != 64:
        raise ValueError("Invalid initial key size")
    
    # Ignore parity bits (PC-1)
    initial_key = permutate(initial_key, PC1_TABLE)

    round_keys = np.empty((16, 48), dtype=int)

    C, D = np.split(initial_key, 2)

    for i in range(16):

        # Just one left shift if in round 1, 2, 9 or 16
        n = 1 if i in [0, 1, 8, 15] else 2

        C = np.roll(C, -n)
        D = np.roll(D, -n)

        # Concatenate and perform PC-2
        round_keys[i] = permutate(np.concatenate((C, D)), PC2_TABLE)

    return round_keys


def encrypt (x, k):

    # Derive round keys
    keys = derive_keys(k)

    # Initial Permutation
    x = permutate(x, IP_TABLE)

    L, R = np.split(x, 2)

    for i in range(16):
        L, R = R, xor(L, f(R, keys[i]))

    x = np.concatenate((R, L))

    # Final Permutation
    y = permutate(x, FP_TABLE)

    return y


def decrypt (y, k):

    # Derive round keys
    keys = derive_keys(k)

    # Initial Permutation
    y = permutate(y, IP_TABLE)

    L, R = np.split(y, 2)

    # Keys are in reverse order
    for i in range(16)[::-1]:
        L, R = R, xor(L, f(R, keys[i]))

    y = np.concatenate((R, L))

    # Final Permutation
    x = permutate(y, FP_TABLE)

    return x


def f (R, k):
    
    # Expansion
    R = permutate(R, E_TABLE)

    R = xor(R, k)

    # Split into eight parts (6 bit)
    R = np.split(R, 8)

    for i in range(8):
        R[i] = substitute(R[i], i)

    # Unite box outputs again to 32 bits
    R = np.concatenate(R)

    # Permutation P
    R = permutate(R, P_TABLE)

    return R


def permutate (x, table):
    """Permutates with given index table"""
    
    y = np.empty_like(table)

    for i, j in np.ndenumerate(table):
        y[i] = x[j - 1] 

    return y


def substitute (x, i):
    """Returns the S-Box output in binary represenation"""

    row = to_decimal(x[::5]) # First and last position
    col = to_decimal(x[1:5]) # Middle four positions

    return to_binary(S_BOX[i][row][col])


def to_decimal (arr):
    """Converts numpy bit vector to integer"""
    return arr.dot(1 << np.arange(arr.shape[-1] - 1, -1, -1))


def to_binary (n):
    """Converts integer to numpy bit vector (4 bit)"""
    return np.array(list(np.binary_repr(n, 4))).astype(np.int8)

    
def xor (a, b):
    """XOR helper function for better readability"""
    return np.logical_xor(a, b) * 1


def encrypt_file (path, k):
    """Encrypts file"""

    # Read file into binary array
    file = read_file(path)

    # Pad with zeros
    while file.size % 64 != 0:
        file = np.append(file, 0)

    # Split into 64 bit blocks
    n = file.size // 64
    file = np.split(file, n)

    # Encrypt each block seperately
    for i in range(n):
        file[i] = encrypt(file[i], k)

    file = np.concatenate(file)

    np.packbits(file).tofile(path)
    return file


def decrypt_file (path, k):
    """Decrypts file"""

    # Read file into binary array
    file = read_file(path)
    
    # Split into 64 bit blocks
    n = file.size // 64
    file = np.split(file, n)

    # Decrypt each block seperately
    for i in range(n):
        file[i] = decrypt(file[i], k)

    file = np.concatenate(file)
    file = np.trim_zeros(file, 'b') # Remove padded zeros

    np.packbits(file).tofile(path)
    return file


def read_file (path):
    """Reads file content into binary numpy array"""
    return np.unpackbits(np.fromfile(path, dtype="uint8"))
