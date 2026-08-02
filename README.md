# Python Extras

Python Extras is a single-file Python library for data structures,
mathematical operations, and escaped-string serialization. Install
`PythonExtras.py` into Python's site-packages directory to import it from any
working directory.

## Features

- Generic storage with `Memory`, `GenObj`, and `readonly`
- Two-dimensional, three-dimensional, and arbitrary-dimensional vectors
- Matrix arithmetic, transposition, determinants, inverses, and vector scaling
- Arithmetic, number theory, trigonometry, calculus, equation solving, and
  numerical helpers through `Calculator`
- Complex-number and time value types
- Delimiter-based string serialization with delimiter and escape preservation
- Tkinter wrappers for applications, forms, scrolling, toolbars, and status bars

`PythonExtras.py` uses NumPy and SciPy for numerical operations. See
[INSTALLATION.md](INSTALLATION.md) for complete manual installation steps.

## Quick Start

```python
from PythonExtras import Calculator, Matrix, Vector, decode, encode

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

## Library Contents

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
| `GUIApp` | Window setup, themes, shortcuts, scheduling, centering, and lifecycle |
| `GUIForm` | Named entry, password, checkbox, and select fields with validation |
| `ScrollableFrame` | A vertically scrollable content container |
| `Toolbar` | Horizontal button, separator, and spacer layout |
| `StatusBar` | Status messages with optional automatic clearing |

### Serialization Functions

`Encode(values, delim="|", escape="\\")` converts an iterable of values into
one escaped string. `Decode(text, delim="|", escape="\\")` restores the list
of strings. Lowercase `encode` and `decode` aliases are also available.

The delimiter and escape marker must each be one character and must be
different. `Decode` rejects truncated escape sequences and input without a
final delimiter.

### GUI Wrappers

The GUI classes wrap standard Tkinter and themed `ttk` widgets. Importing
Python Extras does not create a window; the first window is created only when
`GUIApp` is instantiated.

```python
from PythonExtras import GUIApp, GUIForm, StatusBar, Toolbar

app = GUIApp("Account", size=(520, 320), min_size=(420, 260), theme="clam")
app.content.rowconfigure(1, weight=1)
app.content.columnconfigure(0, weight=1)

toolbar = Toolbar(app.content)
toolbar.grid(row=0, column=0, sticky="ew")

form = GUIForm(app.content, padding=16)
form.grid(row=1, column=0, sticky="nsew")
form.add_entry("name", "Name", required=True)
form.add_password("password", "Password", required=True)
form.add_select("role", ["Reader", "Editor", "Owner"], "Role")
form.add_checkbox("notifications", "Email notifications", default=True)

status = StatusBar(app.content)
status.grid(row=2, column=0, sticky="ew")

def submit():
	try:
		values = form.get_values()
		status.set(f"Saved {values['name']}", clear_after=3000)
	except ValueError as error:
		status.set(str(error))

toolbar.add_button("Save", submit)
toolbar.add_spacer()
toolbar.add_button("Close", app.close)
app.bind_shortcut("<Control-s>", lambda _event: submit())
app.run()
```

For long content, place widgets inside `ScrollableFrame.content`:

```python
from PythonExtras import ScrollableFrame

panel = ScrollableFrame(app.content, padding=12, height=300)
panel.grid(row=1, column=0, sticky="nsew")
# Add child widgets with panel.content as their parent.
```

## More Examples

### Memory Persistence

```python
from PythonExtras import Memory

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
from PythonExtras import Matrix

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
calculator helpers, complex arithmetic, time conversion, and headless GUI
wrapper behavior.

## Project Files

| File | Description |
| --- | --- |
| `PythonExtras.py` | Complete library: structures, math, and serialization |
| `test_python_extras.py` | Automated unit tests |
| `requirements.txt` | NumPy and SciPy dependencies |
| `INSTALLATION.md` | User-wide and system-wide installation instructions |

## Security Note

`Calculator.evalcalc()` and `Calculator.recurrence()` evaluate Python
expressions. Never pass untrusted or user-controlled text to these methods.
