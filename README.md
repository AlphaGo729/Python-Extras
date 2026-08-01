# Python Extras

Python Extras is a small collection of standalone Python modules for data
structures, mathematical operations, and escaped-string serialization. The
files can be used together or copied individually into another Python project.

## Features

- Generic storage with `Memory`, `GenObj`, and `readonly`
- Two-dimensional, three-dimensional, and arbitrary-dimensional vectors
- Matrix arithmetic, transposition, determinants, inverses, and vector scaling
- Arithmetic, number theory, trigonometry, calculus, equation solving, and
	numerical helpers through `Calculator`
- Complex-number and time value types
- Delimiter-based string serialization with delimiter and escape preservation
- No runtime dependency for `Serialization.py`

`MoreStructures.py` uses NumPy and SciPy for numerical operations. See
[INSTALLATION.md](INSTALLATION.md) for complete manual installation steps.

## Quick Start

```python
from MoreStructures import Calculator, Matrix, Vector
from Serialization import decode, encode

calculator = Calculator()
print(calculator.prime_factorization(360))
# {2: 3, 3: 2, 5: 1}

matrix = Matrix([[1, 2], [3, 4]])
vector = Vector([5, 6])
print(matrix.scalevector(vector))
# Vector([17, 39])

values = ["first", "contains|delimiter", r"contains\escape", ""]
encoded = encode(values)
print(decode(encoded))
# ['first', 'contains|delimiter', 'contains\\escape', '']
```

## Modules

### MoreStructures

| Class | Purpose |
| --- | --- |
| `Memory` | Fixed-size addressable storage with JSON file persistence |
| `GenObj` | Small mutable string-backed value container |
| `readonly` | Wrapper exposing a value through `getval()` |
| `Vector2D` | Two-dimensional vector arithmetic, dot products, and cross products |
| `Vector3D` | Three-dimensional vector arithmetic, dot products, and cross products |
| `Vector` | Arbitrary-dimensional vectors and conversion to `Vector2D` or `Vector3D` |
| `Matrix` | Matrix arithmetic, products, transpose, determinant, and inverse |
| `Calculator` | General arithmetic, number theory, numerical methods, and solvers |
| `Complex` | Complex arithmetic, conjugates, and modulus |
| `Time` | Normalized time arithmetic and unit/string conversions |

### Serialization

`Encode(values, delim="|", escape="\\")` converts an iterable of values into
one escaped string. `Decode(text, delim="|", escape="\\")` restores the list
of strings. Lowercase `encode` and `decode` aliases are also available.

The delimiter and escape marker must each be one character and must be
different. `Decode` rejects truncated escape sequences and input without a
final delimiter.

## More Examples

### Memory Persistence

```python
from MoreStructures import Memory

memory = Memory(initialData=[10, "ready", True])
memory.write(0, 20)
memory.filedump("memory.json")

restored = Memory()
restored.fileload("memory.json")
print(restored.data)
# [20, 'ready', True]
```

### Matrix Operations

```python
from MoreStructures import Matrix

matrix = Matrix([[4, 7], [2, 6]])
print(matrix.determinant())
# 10
print(matrix.inverse())
```

## Testing

Run the standard-library test suite from the repository root:

```bash
python3 -m unittest -v
```

The suite covers serialization, memory persistence, vectors, matrices,
calculator helpers, complex arithmetic, and time conversion.

## Project Files

| File | Description |
| --- | --- |
| `MoreStructures.py` | Data structures and mathematical utilities |
| `Serialization.py` | Escaped delimiter-based serialization |
| `test_python_extras.py` | Automated unit tests |
| `requirements.txt` | NumPy and SciPy dependencies |
| `INSTALLATION.md` | Manual installation and removal instructions |

## Security Note

`Calculator.evalcalc()` and `Calculator.recurrence()` evaluate Python
expressions. Never pass untrusted or user-controlled text to these methods.
