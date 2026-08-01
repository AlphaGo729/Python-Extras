# Manual Installation

Python Extras is distributed as one module, `PythonExtras.py`. Installing that
file into a Python site-packages directory makes this import work from any
working directory, like other installed libraries:

```python
from PythonExtras import Calculator, Matrix, encode, decode
```

Python Extras is not currently published on PyPI, so `pip install
PythonExtras` is not available. The module itself must be copied manually.

## Requirements

- Python 3.9 or newer
- `pip`
- NumPy
- SciPy

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

## 2. Install Dependencies

Install NumPy and SciPy for your user account:

```bash
python3 -m pip install --user -r requirements.txt
```

On Windows:

```powershell
py -m pip install --user -r requirements.txt
```

The dependencies and `PythonExtras.py` must be installed for the same Python
interpreter. If your Python installation does not accept `--user`, see
[Virtual Environments](#virtual-environments).

## 3. Install for Your User Account

This is the recommended installation. It does not require administrator access,
and `PythonExtras` will be importable from anywhere for your user account.

### macOS or Linux

Run these commands from the downloaded `Python-Extras` directory:

```bash
USER_SITE="$(python3 -m site --user-site)"
mkdir -p "$USER_SITE"
cp PythonExtras.py "$USER_SITE/PythonExtras.py"
```

Display the location where the module was installed:

```bash
python3 -m site --user-site
```

### Windows PowerShell

Run these commands from the downloaded `Python-Extras` directory:

```powershell
$UserSite = py -m site --user-site
New-Item -ItemType Directory -Force -Path $UserSite
Copy-Item .\PythonExtras.py (Join-Path $UserSite "PythonExtras.py")
```

Display the location where the module was installed:

```powershell
py -m site --user-site
```

### What This Does

Python automatically searches its user site-packages directory when importing
modules. The resulting layout resembles:

```text
site-packages/
`-- PythonExtras.py
```

The exact directory depends on the operating system and Python version. Always
use `python3 -m site --user-site` or `py -m site --user-site` instead of guessing
the path.

## 4. Verify from Another Directory

Change to a directory outside the repository before testing. This confirms
Python is loading the installed copy rather than the local file.

macOS or Linux:

```bash
cd ~
python3 -c "import PythonExtras; print(PythonExtras.__file__)"
python3 -c "from PythonExtras import Calculator; assert Calculator().fibonacci(10) == 55; print('Python Extras is ready')"
```

Windows PowerShell:

```powershell
Set-Location $HOME
py -c "import PythonExtras; print(PythonExtras.__file__)"
py -c "from PythonExtras import Calculator; assert Calculator().fibonacci(10) == 55; print('Python Extras is ready')"
```

The printed path should point to the user site-packages directory, not the
downloaded repository.

You can also verify serialization:

```bash
python3 -c "from PythonExtras import encode, decode; assert decode(encode(['ready'])) == ['ready']; print('Serialization is ready')"
```

## Updating

Download or pull the latest repository version, then repeat the copy command
from step 3. It will replace the installed `PythonExtras.py` file.

For a Git checkout:

```bash
git pull
```

## System-Wide Installation for All Users

A system-wide installation is usually unnecessary. It requires administrator
access and can interfere with an operating-system-managed Python installation.
Prefer the user installation above unless every account on the computer needs
the library.

List the system site-packages directories:

```bash
python3 -c "import site; print(*site.getsitepackages(), sep='\n')"
```

Copy `PythonExtras.py` into the appropriate displayed directory using your
operating system's administrator workflow. Then install NumPy and SciPy for
that same system interpreter. Do not use `sudo pip`; use the system package
manager or a Python installation you manage yourself.

## Virtual Environments

Some managed Python installations disable user-level package installation. In
that case, create a virtual environment and install everything into it:

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp PythonExtras.py "$(python -c 'import site; print(site.getsitepackages()[0])')/PythonExtras.py"
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$VenvSite = python -c "import site; print(site.getsitepackages()[0])"
Copy-Item .\PythonExtras.py (Join-Path $VenvSite "PythonExtras.py")
```

This installation is available from any directory while that virtual
environment is active. It is intentionally isolated from other Python
interpreters.

## Troubleshooting

### `ModuleNotFoundError: No module named 'PythonExtras'`

Confirm that the Python used for installation and execution is the same:

```bash
python3 -c "import sys; print(sys.executable)"
python3 -m site --user-site
```

Then confirm `PythonExtras.py` exists in the displayed user site directory.

### The User Site Directory Is Disabled

Check its status:

```bash
python3 -m site
```

If `ENABLE_USER_SITE` is `False`, use the virtual-environment installation
above or a Python interpreter that enables user site-packages.

### `ModuleNotFoundError: No module named 'numpy'` or `'scipy'`

Install dependencies for the same interpreter used to import Python Extras:

```bash
python3 -m pip install --user -r requirements.txt
```

Using `python3 -m pip` ensures packages are installed for the selected Python
interpreter instead of a different system installation.

### PowerShell Blocks Virtual Environment Activation

Run this command for the current PowerShell process, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Uninstall

Find the installed user directory:

```bash
python3 -m site --user-site
```

Delete `PythonExtras.py` from that directory. You may also delete the matching
`PythonExtras` bytecode file from its `__pycache__` subdirectory. On Windows,
use `py -m site --user-site` to find the directory.

NumPy and SciPy may be shared by other projects, so remove them only when you
know they are no longer needed.