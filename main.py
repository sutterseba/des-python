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
        
if __name__ == "__main__":
    main()
