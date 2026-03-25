import math

pi = 3.142857
m = 8
n = 8

def dctTransform(matrix):

    dct = []
    for i in range(m):
        dct.append([None for _ in range(n)])

    for i in range(m):
        for j in range(n):

            # ci and cj depends on frequency as well as
            # number of row and columns of specified matrix
            if (i == 0):
                ci = 1 / (m ** 0.5)
            else:
                ci = (2 / m) ** 0.5
            if (j == 0):
                cj = 1 / (n ** 0.5)
            else:
                cj = (2 / n) ** 0.5

            # sum will temporarily store the sum of
            # cosine signals
            sum = 0
            for k in range(m):
                for l in range(n):

                    dct1 = matrix[k][l] * math.cos((2 * k + 1) * i * pi / (
                        2 * m)) * math.cos((2 * l + 1) * j * pi / (2 * n))
                    sum = sum + dct1

            dct[i][j] = round(ci * cj * sum, 4)

    for i in range(m):
        for j in range(n):
            print(dct[i][j], end="\t")
        print()

matrix = [[255, 180, 90, 60, 0, 30, 150, 255],
          [100, 200, 255, 80, 40, 130, 75, 220],
          [255, 255, 255, 0, 0, 255, 255, 255],
          [60, 0, 90, 180, 210, 10, 120, 245],
          [10, 20, 30, 40, 120, 220, 180, 100],
          [255, 90, 60, 150, 255, 180, 0, 130],
          [0, 60, 200, 255, 100, 255, 30, 250],
          [30, 130, 255, 180, 75, 90, 255, 200]]

dctTransform(matrix)


# Let we are having a 2-D variable named matrix of dimension 8 X 8 which contains image information and a 2-D variable named dct of same dimension which contain the information after applying discrete cosine transform. So, we have the formula 
# dct[i][j] = ci * cj (sum(k=0 to m-1) sum(l=0 to n-1) matrix[k][l] * cos((2*k+1) *i*pi/2*m) * cos((2*l+1) *j*pi/2*n) 
# where ci= 1/sqrt(m) if i=0 else ci= sqrt(2)/sqrt(m) and 
# similarly, cj= 1/sqrt(n) if j=0 else cj= sqrt(2)/sqrt(n) 
# and we have to apply this formula to all the value, i.e., from i=0 to m-1 and j=0 to n-1
# Here, sum(k=0 to m-1) denotes summation of values from k=0 to k=m-1. 
# In this code, both m and n is equal to 8 and pi is defined as 3.142857.