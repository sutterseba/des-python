import DES
import DES3
import numpy as np


def main ():

    # Random key
    k = DES.new_key()

    # Random message
    x = np.random.randint(2, size=64)

    # Encrypt
    y = DES.encrypt(x, k)

    # Check if decrypted matches original
    print(np.array_equal(x, DES.decrypt(y, k)))

    # Encrypt whole text file
    DES.encrypt_file("test.txt", k)

    # Random 3DES key
    k = DES3.new_key()

    # Random message
    x = np.random.randint(2, size=64)

    # 3DES encrypt
    y = DES3.encrypt(x, k)

    # Check if 3DES decrypted matches original
    print(np.array_equal(x, DES3.decrypt(y, k)))
        
if __name__ == "__main__":
    main()