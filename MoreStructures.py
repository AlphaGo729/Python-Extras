'''
Note to self...
check class Calculator - prime_factorization function to see if it works correctly.



'''
#import modules
import numpy as np
import math
from scipy.optimize import linprog

class Memory:
    '''some RAM, but able to dump to file for hard drive'''
    def __init__(self, size=0, initialData=None):
        '''Initialize memory with a given size and optional initial data.'''
        self.size = size
        self.data = initialData if initialData is not None else [0] * size

    def read(self, address):
        '''Read a value from the specified address in memory.'''
        if 0 <= address < self.size:
            return self.data[address]
        else:
            raise IndexError("Address out of range")
    def write(self, address, value):
        '''Write a value to the specified address in memory.'''
        if 0 <= address < self.size:
            self.data[address] = (value)
        else:
            raise IndexError("Address out of range")
    def filedump(self, filename):
        '''Dump the memory contents to a file.'''
        with open(filename, 'w') as f:
            f.write(self.size)
            for i in range(self.size):
                f.write(f"{i}: {self.data[i]}\n")

    def fileload(self, filename):
        '''Load memory contents from a file.'''
        with open(filename, 'r') as f:
            self.size = int(f.readline().strip())
            for line in f:
                index, value = line.split(': ')
                self.data[int(index)] = int(value.strip())
    


class GenObj:
    '''A generic object that can be used to store any data.'''
    def __init__(self, input):
        '''Initialize the object with input data.'''
        self.data = str(input)
    def __str__(self):
        '''Return a string representation of the object.'''
        return self.data
    def change(self, count):
        '''Change the data stored in the object.'''
        try:
            self.data = str(int(self.data) + count)
        except ValueError:
            return "Invalid data type for change operation"
    def get(self):
        '''Get the data stored in the object.'''
        return self.data
    def set(self, input):
        '''Set the data stored in the object.'''
        self.data = str(input)
    
class readonly:
    '''A read-only datatype that can't be changed.'''
    def __init__(self, val):
        self.val = val
    def getval(self):
        '''Get the value of the read-only attribute.'''
        return self.val
    
class Vector2D:
    '''A simple 2d vector.'''
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector2D(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        return Vector2D(self.x / scalar, self.y / scalar)

    def __repr__(self):
        return f"Vector2D({self.x}, {self.y})"
    def dot(self, other):
        '''Dot product of two vectors.'''
        return self.x * other.x + self.y * other.y
    def cross(self, other):
        '''Cross product of two vectors.'''
        if not isinstance(other, Vector2D):
            raise TypeError("Cross product is only defined between two Vector2D objects")
        return self.x * other.y - self.y * other.x
    
class Vector3D:
    '''A simple 3d vector.'''
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar):
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def __repr__(self):
        return f"Vector3D({self.x}, {self.y}, {self.z})"
    def dot(self, other):
        '''Dot product of two vectors.'''
        return self.x * other.x + self.y * other.y + self.z * other.z
    def cross(self, other):
        '''Cross product of two vectors.'''
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

class Vector:
    '''A simple vector class that can be any dimension'''
    def __init__(self, coord):
        self.coord = coord
        self.numcoord = len(coord)
    def __add__(self, other):
        if self.numcoord != other.numcoord:
            raise ValueError("Vectors must have the same number of dimensions")
        return Vector([self.coord[i] + other.coord[i] for i in range(self.numcoord)])
    def __sub__(self, other):
        if self.numcoord != other.numcoord:
            raise ValueError("Vectors must have the same number of dimensions")
        return Vector([self.coord[i] - other.coord[i] for i in range(self.numcoord)])
    def __mul__(self, scalar):
        return Vector([c * scalar for c in self.coord])
    def __truediv__(self, scalar):
        return Vector([c / scalar for c in self.coord])
    def __repr__(self):
        return self.coord
    def dot(self, other):
        if self.numcoord != other.numcoord:
            raise ValueError("Vectors must have the same number of dimensions")
        return sum(self.coord[i] * other.coord[i] for i in range(self.numcoord))
    

class Matrix:
    '''A simple matrix class that can be any dimension'''
    def __init__(self, rows):
        self.rows = rows
        self.numrows = len(rows)
        self.numcols = len(rows[0]) if rows else 0

    def __add__(self, other):
        if self.numrows != other.numrows or self.numcols != other.numcols:
            raise ValueError("Matrices must have the same dimensions")
        return Matrix([[self.rows[i][j] + other.rows[i][j] for j in range(self.numcols)] for i in range(self.numrows)])

    def __sub__(self, other):
        if self.numrows != other.numrows or self.numcols != other.numcols:
            raise ValueError("Matrices must have the same dimensions")
        return Matrix([[self.rows[i][j] - other.rows[i][j] for j in range(self.numcols)] for i in range(self.numrows)])

    def __mul__(self, scalar):
        return Matrix([[c * scalar for c in row] for row in self.rows])

    def __repr__(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.rows])
    
    def matmul(self, other):
        if self.numcols != other.numrows:
            raise ValueError("Number of columns in the first matrix must match number of rows in the second matrix")
        return Matrix([[sum(self.rows[i][k] * other.rows[k][j] for k in range(self.numcols)) for j in range(other.numcols)] for i in range(self.numrows)])
    def transpose(self):
        return Matrix([[self.rows[j][i] for j in range(self.numrows)] for i in range(self.numcols)])
    def determinant(self):
        if self.numrows != self.numcols:
            raise ValueError("Determinant is only defined for square matrices")
        if self.numrows == 1:
            return self.rows[0][0]
        if self.numrows == 2:
            return self.rows[0][0] * self.rows[1][1] - self.rows[0][1] * self.rows[1][0]
        det = 0
        for c in range(self.numcols):
            submatrix = Matrix([row[:c] + row[c+1:] for row in self.rows[1:]])
            det += ((-1) ** c) * self.rows[0][c] * submatrix.determinant()
        return det
    def inverse(self):
        if self.numrows != self.numcols:
            raise ValueError("Inverse is only defined for square matrices")
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular and cannot be inverted")
        if self.numrows == 1:
            return Matrix([[1 / self.rows[0][0]]])
        cofactors = []
        for i in range(self.numrows):
            cofactor_row = []
            for j in range(self.numcols):
                submatrix = Matrix([row[:j] + row[j+1:] for row in (self.rows[:i] + self.rows[i+1:])])
                cofactor_row.append(((-1) ** (i + j)) * submatrix.determinant())
            cofactors.append(cofactor_row)
        cofactors_transposed = Matrix(cofactors).transpose()
        return cofactors_transposed * (1 / det)
    def scalevector(self, vector):
        if self.numcols != len(vector.coord):
            raise ValueError("Matrix columns must match vector dimensions")
        return Vector([sum(self.rows[i][j] * vector.coord[j] for j in range(self.numcols)) for i in range(self.numrows)])
        

class Calculator:
    '''A simple calculator class that can perform basic arithmetic operations.'''
    def __init__(self):
        self.memory = 0

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b

    def truediv(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    def set_memory(self, value):
        self.memory = value

    def get_memory(self):
        return self.memory
    def clear_memory(self):
        self.memory = 0
    def evalcalc(self, expression):
        '''Evaluate a mathematical expression.'''
        try:
            return eval(expression)
        except Exception as e:
            return f"Error evaluating expression: {e}"
    def factorial(self, n):
        '''Calculate the factorial of a number.'''
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
    def power(self, base, exponent):
        '''Calculate the power of a number.'''
        return base ** exponent
    def rootf(self, base, root=2):
        '''Calculate the nth root of a number.'''
        if root == 0:
            raise ValueError("Cannot calculate the root with zero")
        return base ** (1 / root)
    def logarithm(self, value, base=10):
        '''Calculate the logarithm of a number.'''
        if value <= 0:
            raise ValueError("Logarithm is not defined for non-positive values")
        if base <= 1:
            raise ValueError("Base must be greater than 1")
        return math.log(value, base)
    def gcd2(self, a, b):
        '''Calculate the greatest common divisor of two numbers.'''
        while b:
            a, b = b, a % b
        return abs(a)
    def lcm2(self, a, b):
        '''Calculate the least common multiple of two numbers.'''
        if a == 0 or b == 0:
            return 0
        return abs(a * b) // self.gcd2(a, b)
    def gcd(self, *args):
        '''Calculate the greatest common divisor of multiple numbers.'''
        if len(args) < 2:
            raise ValueError("At least two numbers are required")
        gcd_value = args[0]
        for num in args[1:]:
            gcd_value = self.gcd2(gcd_value, num)
        return gcd_value
    def lcm(self, *args):
        '''Calculate the least common multiple of multiple numbers.'''
        if len(args) < 2:
            raise ValueError("At least two numbers are required")
        lcm_value = args[0]
        for num in args[1:]:
            lcm_value = self.lcm2(lcm_value, num)
        return lcm_value
    def is_prime(self, n):
        '''Check if a number is prime.'''
        if n <= 1:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    def prime_factors(self, n):
        '''Find the prime factors of a number.'''
        if n <= 1:
            return []
        factors = []
        for i in range(2, int(n**0.5) + 1):
            while n % i == 0:
                factors.append(i)
                n //= i
        if n > 1:
            factors.append(n)
        return factors
    def fibonacci(self, n):
        '''Generate Fibonacci sequence up to n terms.'''
        if n <= 0:
            return []
        fib_sequence = [0, 1]
        while len(fib_sequence) < n:
            fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
        return fib_sequence[n]
    def recurrence(self, n, base_cases, recurrence_func_str):
        memo = dict(base_cases)
        results = []

        def helper(k):
            if k in memo:
                return memo[k]
            memo[k] = eval(recurrence_func_str, {"k": k, "f": helper})
            return memo[k]

        for i in range(n + 1):
            results.append(helper(i))

        return results
    def sin(self, angle):
        '''Calculate the sine of an angle in degrees.'''
        return math.sin(math.radians(angle))
    def cos(self, angle):
        '''Calculate the cosine of an angle in degrees.'''
        return math.cos(math.radians(angle))
    def tan(self, angle):
        '''Calculate the tangent of an angle in degrees.'''
        return math.tan(math.radians(angle))
    def asin(self, value):
        '''Calculate the arcsine of a value.'''
        if -1 <= value <= 1:
            return math.degrees(math.asin(value))
        else:
            raise ValueError("Value must be in the range [-1, 1]")
    def acos(self, value):
        '''Calculate the arccosine of a value.'''
        if -1 <= value <= 1:
            return math.degrees(math.acos(value))
        else:
            raise ValueError("Value must be in the range [-1, 1]")
    def atan(self, value):
        '''Calculate the arctangent of a value.'''
        return math.degrees(math.atan(value))
    def atan2(self, y, x):
        '''Calculate the arctangent of y/x, handling the quadrant correctly.'''
        return math.degrees(math.atan2(y, x))
    def sec(self, angle):
        '''Calculate the secant of an angle in degrees.'''
        if self.cos(angle) == 0:
            raise ValueError("Secant is undefined for angles where cosine is zero")
        return 1 / self.cos(angle)
    def csc(self, angle):
        '''Calculate the cosecant of an angle in degrees.'''
        if self.sin(angle) == 0:
            raise ValueError("Cosecant is undefined for angles where sine is zero")
        return 1 / self.sin(angle)
    def cot(self, angle):
        '''Calculate the cotangent of an angle in degrees.'''
        if self.tan(angle) == 0:
            raise ValueError("Cotangent is undefined for angles where tangent is zero")
        return 1 / self.tan(angle)
    def secant(self, angle):
        '''Calculate the secant of an angle in degrees.'''
        if self.cos(angle) == 0:
            raise ValueError("Secant is undefined for angles where cosine is zero")
        return 1 / self.cos(angle)
    def cosecant(self, angle):
        '''Calculate the cosecant of an angle in degrees.'''
        if self.sin(angle) == 0:
            raise ValueError("Cosecant is undefined for angles where sine is zero")
        return 1 / self.sin(angle)
    def cotangent(self, angle):
        '''Calculate the cotangent of an angle in degrees.'''
        if self.tan(angle) == 0:
            raise ValueError("Cotangent is undefined for angles where tangent is zero")
        return 1 / self.tan(angle)
    def hyperbolic_sine(self, x):
        '''Calculate the hyperbolic sine of x.'''
        return (math.exp(x) - math.exp(-x)) / 2
    def hyperbolic_cosine(self, x):
        '''Calculate the hyperbolic cosine of x.'''
        return (math.exp(x) + math.exp(-x)) / 2
    def hyperbolic_tangent(self, x):
        '''Calculate the hyperbolic tangent of x.'''
        return self.hyperbolic_sine(x) / self.hyperbolic_cosine(x)
    def hyperbolic_arcsine(self, x):
        '''Calculate the hyperbolic arcsine of x.'''
        return math.log(x + math.sqrt(x**2 + 1))
    def hyperbolic_arccosine(self, x):
        '''Calculate the hyperbolic arccosine of x.'''
        if x < 1:
            raise ValueError("Value must be greater than or equal to 1")
        return math.log(x + math.sqrt(x**2 - 1))
    def hyperbolic_arctangent(self, x):
        '''Calculate the hyperbolic arctangent of x.'''
        if abs(x) >= 1:
            raise ValueError("Value must be in the range (-1, 1)")
        return 0.5 * math.log((1 + x) / (1 - x))
    def hyperbolic_arccosecant(self, x):
        '''Calculate the hyperbolic arccosecant of x.'''
        if abs(x) < 1:
            raise ValueError("Value must be greater than or equal to 1 or less than or equal to -1")
        return math.log(math.sqrt(x**2 + 1) + x)
    def hyperbolic_arcsecant(self, x):
        '''Calculate the hyperbolic arcsecant of x.'''
        if abs(x) < 1:
            raise ValueError("Value must be greater than or equal to 1 or less than or equal to -1")
        return math.log(math.sqrt(x**2 - 1) + x)
    def hyperbolic_arccotangent(self, x):
        '''Calculate the hyperbolic arccotangent of x.'''
        if x == 0:
            raise ValueError("Value cannot be zero")
        return 0.5 * math.log((x + 1) / (x - 1))
    def hyperbolic_secant(self, x):
        '''Calculate the hyperbolic secant of x.'''
        return 2 / (math.exp(x) + math.exp(-x))
    def hyperbolic_cosecant(self, x):
        '''Calculate the hyperbolic cosecant of x.'''
        if self.hyperbolic_sine(x) == 0:
            raise ValueError("Hyperbolic cosecant is undefined for x where hyperbolic sine is zero")
        return 2 / (math.exp(x) - math.exp(-x))
    def hyperbolic_cotangent(self, x):
        '''Calculate the hyperbolic cotangent of x.'''
        if self.hyperbolic_sine(x) == 0:
            raise ValueError("Hyperbolic cotangent is undefined for x where hyperbolic sine is zero")
        return self.hyperbolic_cosine(x) / self.hyperbolic_sine(x)
    def comparator(self, a, b):
        '''Compare two values and return -1, 0, or 1.'''
        if a < b:
            return -1
        elif a > b:
            return 1
        else:
            return 0
    def is_even(self, n):
        '''Check if a number is even.'''
        return n % 2 == 0
    def is_odd(self, n):
        '''Check if a number is odd.'''
        return n % 2 != 0
    def is_palindrome(self, s):
        '''Check if a string is a palindrome.'''
        s = s.lower().replace(" ", "")
        return s == s[::-1]
    def is_anagram(self, s1, s2):
        '''Check if two strings are anagrams of each other.'''
        s1 = s1.lower().replace(" ", "")
        s2 = s2.lower().replace(" ", "")
        return sorted(s1) == sorted(s2)
    def is_substring(self, s1, s2):
        '''Check if s1 is a substring of s2.'''
        return s1 in s2
    def is_prime_factor(self, n, factor):
        '''Check if a number is a prime factor of another number.'''
        if n <= 1 or factor <= 1:
            return False
        if n % factor != 0:
            return False
        for i in range(2, int(factor**0.5) + 1):
            if factor % i == 0:
                return False
        return True
    def is_perfect_square(self, n):
        '''Check if a number is a perfect square.'''
        if n < 0:
            return False
        root = int(n**0.5)
        return root * root == n
    def is_fibonacci(self, n):
        '''Check if a number is a Fibonacci number.'''
        if n < 0:
            return False
        a, b = 0, 1
        while a < n:
            a, b = b, a + b
        return a == n
    def is_perfect_number(self, n):
        '''Check if a number is a perfect number.'''
        if n <= 0:
            return False
        divisors_sum = sum(i for i in range(1, n) if n % i == 0)
        return divisors_sum == n
    def is_perfect_power(self, n,power=2):
        '''Check if a number is a perfect power.'''
        if n < 1:
            return False
        for base in range(2, int(n**0.5) + 1):
            power = 2
            while base ** power <= n:
                if base ** power == n:
                    return True
                power += 1
        return False
    def is_armstrong(self, n):
        '''Check if a number is an Armstrong number.'''
        digits = str(n)
        num_digits = len(digits)
        return sum(int(digit) ** num_digits for digit in digits) == n
    def is_abundant(self, n):
        '''Check if a number is an abundant number.'''
        if n <= 0:
            return False
        divisors_sum = sum(i for i in range(1, n) if n % i == 0)
        return divisors_sum > n
    def is_deficient(self, n):
        '''Check if a number is a deficient number.'''
        if n <= 0:
            return False
        divisors_sum = sum(i for i in range(1, n) if n % i == 0)
        return divisors_sum < n
    def is_semiperfect(self, n):
        '''Check if a number is a semiperfect number.'''
        if n <= 0:
            return False
        divisors = [i for i in range(1, n) if n % i == 0]
        return self.is_subset_sum(divisors, n)
    def is_subset_sum(self, nums, target):
        '''Check if there is a subset of nums that sums to target.'''
        n = len(nums)
        dp = [[False] * (target + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = True
        for i in range(1, n + 1):
            for j in range(1, target + 1):
                if nums[i - 1] <= j:
                    dp[i][j] = dp[i - 1][j] or dp[i - 1][j - nums[i - 1]]
                else:
                    dp[i][j] = dp[i - 1][j]
        return dp[n][target]
    def is_square_free(self, n):
        '''Check if a number is square-free.'''
        if n < 1:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % (i * i) == 0:
                return False
        return True
    def is_circular_prime(self, n):
        '''Check if a number is a circular prime.'''
        if n < 2:
            return False
        str_n = str(n)
        for i in range(len(str_n)):
            rotated = int(str_n[i:] + str_n[:i])
            if not self.is_prime(rotated):
                return False
        return True
    def is_truncatable_prime(self, n):
        '''Check if a number is a truncatable prime.'''
        if n < 10:
            return False
        str_n = str(n)
        for i in range(len(str_n)):
            if not self.is_prime(int(str_n[i:])) or not self.is_prime(int(str_n[:i + 1])):
                return False
        return True
    def is_lychrel(self, n, max_iterations=50):
        '''Check if a number is a Lychrel number.'''
        if n < 0:
            return False
        for _ in range(max_iterations):
            n += int(str(n)[::-1])
            if self.is_palindrome(str(n)):
                return False
        return True
    def is_narcissistic(self, n):
        '''Check if a number is a narcissistic number.'''
        digits = str(n)
        num_digits = len(digits)
        return sum(int(digit) ** num_digits for digit in digits) == n
    def is_smith(self, n):
        '''Check if a number is a Smith number.'''
        if n < 2 or self.is_prime(n):
            return False
        digit_sum = sum(int(digit) for digit in str(n))
        prime_factor_sum = sum(self.prime_factors(n))
        return digit_sum == prime_factor_sum
    def is_automorphic(self, n):
        '''Check if a number is an automorphic number.'''
        square = n * n
        return str(square).endswith(str(n))
    def is_pandigital(self, n):
        '''Check if a number is pandigital.'''
        digits = str(n)
        return set(digits) == set(str(i) for i in range(len(digits))) and len(digits) == len(set(digits))
    def is_fascinating(self, n):
        '''Check if a number is fascinating.'''
        if n < 100:
            return False
        concatenated = str(n) + str(n * 2) + str(n * 3)
        return set(concatenated) == set('123456789') and len(concatenated) == 9
    def is_harshad(self, n):
        '''Check if a number is a Harshad number.'''
        if n <= 0:
            return False
        digit_sum = sum(int(digit) for digit in str(n))
        return n % digit_sum == 0
    def is_kaprekar(self, n):
        '''Check if a number is a Kaprekar number.'''
        if n < 0:
            return False
        square = n * n
        str_square = str(square)
        d = len(str(n))
        left_part = int(str_square[:-d]) if str_square[:-d] else 0
        right_part = int(str_square[-d:])
        return left_part + right_part == n
    def is_amicable(self, a, b):
        '''Check if two numbers are amicable.'''
        def sum_of_divisors(n):
            return sum(i for i in range(1, n) if n % i == 0)
        return sum_of_divisors(a) == b and sum_of_divisors(b) == a and a != b
    def derivative(self, func, x, h=1e-5):
        '''Calculate the derivative of a function at a point x using finite difference.'''
        return (func(x + h) - func(x - h)) / (2 * h)
    def integral(self, func, a, b, n=1000):
        '''Calculate the definite integral of a function from a to b using the trapezoidal rule.'''
        h = (b - a) / n
        integral_value = 0.5 * (func(a) + func(b))
        for i in range(1, n):
            integral_value += func(a + i * h)
        return integral_value * h
    def solve_quadratic(self, a, b, c):
        '''Solve a quadratic equation ax^2 + bx + c = 0.'''
        if a == 0:
            raise ValueError("Coefficient 'a' cannot be zero")
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return None
        elif discriminant == 0:
            return -b / (2 * a)
        else:
            root1 = (-b + discriminant**0.5) / (2 * a)
            root2 = (-b - discriminant**0.5) / (2 * a)
            return (root1, root2)
    def solve_linear(self, a, b):
        '''Solve a linear equation ax + b = 0.'''
        if a == 0:
            raise ValueError("Coefficient 'a' cannot be zero")
        return -b / a
    def solve_system_of_equations(self, equations):
        '''Solve a system of linear equations using matrix methods.'''
        A = np.array([[eq[0] for eq in equations], [eq[1] for eq in equations]])
        b = np.array([eq[2] for eq in equations])
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return "No unique solution exists"
    def solve_polynomial(self, coefficients):
        '''Solve a polynomial equation with given coefficients.'''
        return np.roots(coefficients)
    def solve_differential_equation(self, func, y0, t0, t1, n=1000):
        '''Solve a first-order differential equation using Euler's method.'''
        h = (t1 - t0) / n
        t = t0
        y = y0
        results = [(t, y)]
        for _ in range(n):
            y += h * func(t, y)
            t += h
            results.append((t, y))
        return results
    def solve_ode(self, func, y0, t0, t1, n=1000):
        '''Solve an ordinary differential equation using the Runge-Kutta method.'''
        h = (t1 - t0) / n
        t = t0
        y = y0
        results = [(t, y)]
        
        for _ in range(n):
            k1 = h * func(t, y)
            k2 = h * func(t + h / 2, y + k1 / 2)
            k3 = h * func(t + h / 2, y + k2 / 2)
            k4 = h * func(t + h, y + k3)
            y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            t += h
            results.append((t, y))
        
        return results
    def solve_pde(self, func, x0, x1, y0, y1, n=100, m=100):
        '''Solve a partial differential equation using finite difference method.'''
        x = np.linspace(x0, x1, n)
        y = np.linspace(y0, y1, m)
        dx = (x1 - x0) / (n - 1)
        dy = (y1 - y0) / (m - 1)
        
        grid = np.zeros((n, m))
        
        for i in range(n):
            for j in range(m):
                grid[i][j] = func(x[i], y[j])
        
        for _ in range(100):
            new_grid = grid.copy()
            for i in range(1, n - 1):
                for j in range(1, m - 1):
                    new_grid[i][j] = (grid[i + 1][j] + grid[i - 1][j] + grid[i][j + 1] + grid[i][j - 1]) / 4
            grid = new_grid
        return grid
    def solve_linear_programming(self, c, A, b):
        '''Solve a linear programming problem using the simplex method.'''
        res = linprog(c, A_ub=A, b_ub=b)
        if res.success:
            return res.x, res.fun
        else:
            return "No solution found"
    def find_inverse_modulo(self, a, m):
        '''Find the modular inverse of a under modulo m using the Extended Euclidean Algorithm.'''
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        gcd, x, _ = extended_gcd(a, m)
        if gcd != 1:
            raise ValueError("Inverse does not exist")
        else:
            return x % m
    def solve_diophantine(self, a, b, c):
        '''Solve a linear Diophantine equation ax + by = c.'''
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        gcd, x0, y0 = extended_gcd(a, b)
        if c % gcd != 0:
            return "No integer solutions exist"
        else:
            x0 *= c // gcd
            y0 *= c // gcd
            return x0, y0
    def base_conv(self,number,base1,base2):
        '''Convert a number from base1 to base2.'''
        if base1 < 2 or base2 < 2:
            raise ValueError("Base must be at least 2")
        if isinstance(number, str):
            number = int(number, base1)
        elif not isinstance(number, int):
            raise TypeError("Number must be an integer or a string representing an integer")
        if number < 0:
            raise ValueError("Number must be non-negative")
        
        if number == 0:
            return '0'
        
        digits = []
        while number > 0:
            digits.append(str(number % base2))
            number //= base2
        return ''.join(digits[::-1])
    
    def logicgates(self,inputs,gate):
        '''Simulate basic logic gates.'''
        if gate == 'AND':
            return all(inputs)
        elif gate == 'OR':
            return any(inputs)
        elif gate == 'NOT':
            return [not (inputs[i]) for i in range(len(inputs))]
        elif gate == 'NAND':
            return not all(inputs)
        elif gate == 'NOR':
            return not any(inputs)
        elif gate == 'XOR':
            return inputs[0] != inputs[1]
        elif gate == 'XNOR':
            return inputs[0] == inputs[1]
        else:
            raise ValueError("Unsupported gate type")
    def set_precision(self, precision):
        '''Set the precision for floating-point calculations.'''
        import decimal
        decimal.getcontext().prec = precision
    def additive_inverse(self, a):
        '''Calculate the additive inverse of a number.'''
        return -a
    def multiplicative_inverse(self, a):
        '''Calculate the multiplicative inverse of a number.'''
        if a == 0:
            raise ValueError("Multiplicative inverse is undefined for zero")
        return 1 / a
    def absolute_value(self, a):
        '''Calculate the absolute value of a number.'''
        return abs(a)
    def signum(self, a):
        '''Calculate the signum function of a number.'''
        if a > 0:
            return 1
        elif a < 0:
            return -1
        else:
            return 0
    def floor(self, a):
        '''Calculate the floor of a number.'''
        return math.floor(a)
    def ceil(self, a):
        '''Calculate the ceiling of a number.'''
        return math.ceil(a)
    def round(self, a, ndigits=0):
        '''Round a number to a specified number of decimal places.'''
        return round(a, ndigits)
    def find_roots_of_unity(self, n):
        '''Find the nth roots of unity.'''
        if n <= 0:
            raise ValueError("n must be a positive integer")
        roots = []
        for k in range(n):
            angle = 2 * 3.141592653589793 * k / n
            roots.append(Complex(math.cos(angle), math.sin(angle)))
        return roots
    def find_euler_totient(self, n):
        '''Calculate Euler's totient function for a number n.'''
        if n < 1:
            raise ValueError("n must be a positive integer")
        count = 0
        for i in range(1, n + 1):
            if self.gcd2(n, i) == 1:
                count += 1
        return count
    def find_divisors(self, n):
        '''Find all divisors of a number n.'''
        if n < 1:
            raise ValueError("n must be a positive integer")
        divisors = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divisors.append(i)
                if i != n // i:
                    divisors.append(n // i)
        return sorted(divisors)
    def prime_sieve(self, limit):
        '''Generate all prime numbers up to a given limit using the Sieve of Eratosthenes.'''
        if limit < 2:
            return []
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i in range(limit + 1) if sieve[i]]
    def prime_factorization(self, n):
        '''Find the prime factorization of a number n.'''
        if n < 2:
            raise ValueError("n must be a positive integer greater than 1")
        prime_factors = []
        powers = []
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                count = 0
                while n % i == 0:
                    n //= i
                    count += 1
                prime_factors.append(i)
                powers.append(count)
        if n > 1 and len(prime_factors) == 0:
            prime_factors.append(n)
            powers.append(1)
        return dict(zip(prime_factors, powers))
    




class Complex:
    '''A simple complex number class.'''
    def __init__(self, real=0, imag=0):
        self.real = real
        self.imag = imag
    
    def __eq__(self, other):
        return self.real == other.real and self.imag == other.imag
    
    def __str__(self):
        return f"{self.real} + {self.imag}i" if self.imag >= 0 else f"{self.real} - {-self.imag}i"

    def __add__(self, other):
        return Complex(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other):
        return Complex(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other):
        return Complex(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real
        )

    def __truediv__(self, other):
        denom = other.real**2 + other.imag**2
        if denom == 0:
            raise ZeroDivisionError("Division by zero in complex division")
        return Complex(
            (self.real * other.real + self.imag * other.imag) / denom,
            (self.imag * other.real - self.real * other.imag) / denom
        )

    def __repr__(self):
        return f"Complex({self.real}, {self.imag})"
    
    def conjugate(self):
        '''Return the complex conjugate.'''
        return Complex(self.real, -self.imag)
    
    def modulus(self):
        '''Return the modulus of the complex number.'''
        return (self.real**2 + self.imag**2)**0.5




class Time:
    '''A simple time class that can handle hours, minutes, and seconds.'''
    def __init__(self, hours=0, minutes=0, seconds=0):
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds
        self.normalize()

    def normalize(self):
        '''Normalize the time to ensure valid values.'''
        if self.seconds >= 60:
            self.minutes += self.seconds // 60
            self.seconds %= 60
        if self.minutes >= 60:
            self.hours += self.minutes // 60
            self.minutes %= 60
        if self.hours < 0 or self.minutes < 0 or self.seconds < 0:
            raise ValueError("Time cannot be negative")
    
    def __str__(self):
        return f"{self.hours:02}:{self.minutes:02}:{self.seconds:02}"
    
    def __add__(self, other):
        return Time(
            self.hours + other.hours,
            self.minutes + other.minutes,
            self.seconds + other.seconds
        )
    
    def __sub__(self, other):
        return Time(
            self.hours - other.hours,
            self.minutes - other.minutes,
            self.seconds - other.seconds
        )
    def __mul__(self, scalar):
        total_seconds = self.to_seconds() * scalar
        return Time.from_seconds(total_seconds)
    def __truediv__(self, scalar):
        if scalar == 0:
            raise ValueError("Cannot divide by zero")
        total_seconds = self.to_seconds() / scalar
        return Time.from_seconds(total_seconds)
    def to_seconds(self):
        '''Convert the time to total seconds.'''
        return self.hours * 3600 + self.minutes * 60 + self.seconds
    @classmethod
    def from_seconds(cls, total_seconds):
        '''Create a Time object from total seconds.'''
        if total_seconds < 0:
            raise ValueError("Total seconds cannot be negative")
        hours = total_seconds // 3600
        total_seconds %= 3600
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return cls(hours, minutes, seconds)
    def to_minutes(self):
        '''Convert the time to total minutes.'''
        return self.hours * 60 + self.minutes + self.seconds / 60
    @classmethod
    def from_minutes(cls, total_minutes):
        '''Create a Time object from total minutes.'''
        if total_minutes < 0:
            raise ValueError("Total minutes cannot be negative")
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return cls(hours, minutes, 0)
    def to_hours(self):
        '''Convert the time to total hours.'''
        return self.hours + self.minutes / 60 + self.seconds / 3600
    @classmethod
    def from_hours(cls, total_hours):
        '''Create a Time object from total hours.'''
        if total_hours < 0:
            raise ValueError("Total hours cannot be negative")
        hours = int(total_hours)
        minutes = int((total_hours - hours) * 60)
        seconds = int(((total_hours - hours) * 60 - minutes) * 60)
        return cls(hours, minutes, seconds)
    def to_string(self):
        '''Return the time as a formatted string.'''
        return f"{self.hours:02}:{self.minutes:02}:{self.seconds:02}"
    @classmethod
    def from_string(cls, time_str):
        '''Create a Time object from a formatted string.'''
        parts = time_str.split(':')
        if len(parts) != 3:
            raise ValueError("Time string must be in the format HH:MM:SS")
        hours, minutes, seconds = map(int, parts)
        return cls(hours, minutes, seconds)
    def get_current_time(self):
        '''Get the current time as a Time object.'''
        from datetime import datetime
        now = datetime.now()
        return Time(now.hour, now.minute, now.second)
