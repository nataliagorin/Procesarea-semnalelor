import numpy
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

from matplotlib.image import imread

## ex 1
a = 2
b = 3
print(a + b)

c = a + b
print(c)

print(a*b)
print(a**b)


## ex2
vec1 = [1, 2, 3, 4]
print(vec1)

vec2 = np.array([1, 2, 3, 4])
print(vec2)

vec3 = np.ones(7, int)
print(vec3)

vec4 = np.random.randn(5)
print(vec4)

print(vec1[0])

print(vec1[1:3])

print(vec1[-2:])

## ex3
x = [[1, 2], [4, 5], [3, 6]]
print(x)

print(x[0])
print(x[0][0])

mat = np.matrix(np.random.rand(4, 5))

print(mat)

## ex 4
matTrans = numpy.transpose(mat)
print("mat_Trans", matTrans)

matOnes = np.ones((4, 1), int)
print("mat_Ones", matOnes)

matones_trans = numpy.transpose(matOnes)
print(matones_trans)


## ex 5
v = [x for x in range(5, 15)]
print(v)

ve = [x for x in range(1, 10, 3)]
print(ve)

n = numpy.linspace(1, 10, 3)
print(n)

len_vec = len(v)
len_matrix = mat.shape
print(len_vec, len_matrix)

## ex 6
s = 'Hello'
print(s)


str = "Signal processing"
print(str)

str1 = "Lab of "
print(str1 + str)

f = 3.14
print(f"Pi is {f}")


## ex 7
freq = 1
timp = np.linspace(0, 1, 100)
sinus = np.sin(2 * np.pi * freq * timp)

plt.plot(timp, sinus, color='b', linestyle='--', label='sinusoida de frecventa 1 Hz')
plt.legend()


freq1 = 2
timp1 = np.linspace(0, 1, 100)
sinus1 = np.sin(2 * np.pi * freq1 * timp1)

plt.plot(timp1, sinus1, color='g', label='sinusoida de frecventa 2 Hz')
plt.legend()

plt.title('Sinus')
plt.xlabel('Timp [s]')
plt.ylabel('Amplitudine')


plt.show()

## ex 8
freq2 = 3
freq3 = 4
timp2 = np.linspace(0, 1, 100)

s1 = np.sin(2 * np.pi * freq2 * timp2)
s2 = np.sin(2 * np.pi * freq3 * timp2)

plt.figure()
plt.plot(timp2, s1, color='b', label='sinusoida de frecventa 3 Hz')
plt.legend()

plt.figure()
plt.plot(timp2, s2, color='g', label='sinusoida de frecventa 4 Hz')
plt.legend()

sum_sin = sinus + sinus1
plt.figure()
plt.plot(timp, sum_sin, color='r', label='suma sinusurilor')
plt.legend()
plt.show()


## ex 9
IR = imread('IR.png')
plt.figure()
plt.imshow(IR)
plt.show()

R1 = imread('R1.png')
plt.figure()
plt.imshow(R1)
plt.show()

R2 = imread('R2.png')
plt.figure()
plt.imshow(R2)
plt.show()




init = (IR - R1 * 0.3 - R2 *0.3)/0.3
plt.imshow(init)
plt.show()



## ex 10
N = 1000
A = np.random.rand(N, N)
B = np.random.rand(N, N)
C = np.zeros((N, N))

time1 = time.time()
for i in range(N):
    for j in range(N):
        C[i, j] = A[i, j] * B[i, j]

time_row = time.time() - time1

time2 = time.time()
for j in range(N):
    for i in range(N):
        C[i, j] = A[i, j] * B[i, j]
time_column = time.time() - time2

time3 = time.time()
C = A * B
time_matrix_operation = time.time() - time3

print(f'Timp row-major: {time_row} s')
print(f'Timp column-major: {time_column} s')
print(f'Timp matrix operation: {time_matrix_operation} s')