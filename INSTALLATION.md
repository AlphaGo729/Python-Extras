# Manual Installation

Python Extras currently consists of standalone Python modules. It is not a
published Python package, so install it manually by downloading the repository
and either running from its directory or copying the modules into your project.

## Requirements

- Python 3.9 or newer
- `pip`
- NumPy and SciPy when using `MoreStructures.py`

`Serialization.py` uses only the Python standard library and can be installed
without NumPy or SciPy.

Check that Python and pip are available:

```bash
python3 --version
python3 -m pip --version
```

On Windows, use `py` instead of `python3` if that is how Python is installed.

## 1. Download the Files

### Git

```bash
git clone https://github.com/AlphaGo729/Python-Extras.git
cd Python-Extras
```

To update this copy later:

```bash
git pull
```

### ZIP Download

1. Open <https://github.com/AlphaGo729/Python-Extras>.
2. Select **Code**, then **Download ZIP**.
3. Extract the archive.
4. Open a terminal in the extracted `Python-Extras` directory.

## 2. Create a Virtual Environment

A virtual environment keeps the numerical dependencies separate from the rest
of your Python installation.

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```batch
py -m venv .venv
.venv\Scripts\activate.bat
```

The active terminal normally shows `(.venv)` before the prompt.

## 3. Install Dependencies

With the virtual environment active, install the dependencies used by
`MoreStructures.py`:

```bash
python3 -m pip install -r requirements.txt
```

On Windows:

```powershell
py -m pip install -r requirements.txt
```

Skip this step when using only `Serialization.py`.

## 4. Make the Modules Available

Choose one of the following approaches.

### Run Inside the Repository

Place your script in the repository directory, next to `MoreStructures.py` and
`Serialization.py`. Python can then import the modules directly:

```python
from MoreStructures import Calculator, Vector
from Serialization import decode, encode
```

### Copy Modules Into an Existing Project

Copy the module files you need into the same directory as your application:

```text
your-project/
|-- app.py
|-- MoreStructures.py
`-- Serialization.py
```

Install NumPy and SciPy in that project's environment when copying
`MoreStructures.py`:

```bash
python3 -m pip install numpy scipy
```

You may copy only `Serialization.py` when serialization is the only feature you
need.

## 5. Verify the Installation

From the directory containing the modules, run:

```bash
python3 -c "from MoreStructures import Calculator; assert Calculator().fibonacci(10) == 55; print('MoreStructures is ready')"
python3 -c "from Serialization import encode, decode; assert decode(encode(['ready'])) == ['ready']; print('Serialization is ready')"
```

On Windows, replace `python3` with `py` when necessary.

For a full verification from the cloned repository:

```bash
python3 -m unittest -v
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'MoreStructures'`

Run the script from the directory containing `MoreStructures.py`, or copy the
file into the same project directory as the script importing it.

### `ModuleNotFoundError: No module named 'numpy'` or `'scipy'`

Activate the same virtual environment used to run the application, then run:

```bash
python3 -m pip install -r requirements.txt
```

Using `python3 -m pip` ensures packages are installed for the selected Python
interpreter instead of a different system installation.

### PowerShell Blocks Virtual Environment Activation

Run this command for the current PowerShell process, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Uninstall

Remove any copied `MoreStructures.py` and `Serialization.py` files from the
consuming project. If you created a dedicated clone and virtual environment,
delete the cloned `Python-Extras` directory. This does not affect other Python
environments.