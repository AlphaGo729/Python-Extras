# Python Extras

A small collection of reusable Python data structures, mathematical helpers,
and escaped-string serialization utilities.

## Setup

Python 3.9 or newer is recommended.

```bash
python3 -m pip install -r requirements.txt
```

## Usage

```python
from MoreStructures import Calculator, Matrix, Vector
from Serialization import Decode, Encode

calculator = Calculator()
print(calculator.prime_factorization(360))

matrix = Matrix([[1, 2], [3, 4]])
vector = Vector([5, 6])
print(matrix.scalevector(vector))

encoded = Encode(["first", "contains|delimiter", r"contains\escape"])
print(Decode(encoded))
```

`MoreStructures.py` includes memory storage, vectors, matrices, complex numbers,
time values, and calculator functions. `Serialization.py` provides `Encode` and
`Decode`, with lowercase aliases for conventional Python naming.

## Tests

Run the complete test suite from the repository root:

```bash
python3 -m unittest -v
```
