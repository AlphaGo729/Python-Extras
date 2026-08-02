import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest import mock

import PythonExtras
from PythonExtras import Calculator, Complex, GUIApp, GUIForm, Matrix, Memory, Time, Vector, Vector2D, Vector3D


class SerializationTests(unittest.TestCase):
    def test_round_trip_preserves_delimiters_escapes_and_empty_values(self):
        values = ["plain", "a|b", r"c\d", ""]

        encoded = PythonExtras.Encode(values)

        self.assertEqual(PythonExtras.Decode(encoded), values)

    def test_custom_characters_and_lowercase_aliases(self):
        values = ["one;two", "three!four"]

        encoded = PythonExtras.encode(values, delim=";", escape="!")

        self.assertEqual(PythonExtras.decode(encoded, delim=";", escape="!"), values)

    def test_decode_rejects_incomplete_data(self):
        with self.assertRaises(ValueError): 
            PythonExtras.Decode("value")
        with self.assertRaises(ValueError):
            PythonExtras.Decode("value\\")


class MemoryTests(unittest.TestCase):
    def test_read_write_and_bounds(self):
        memory = Memory(2)
        memory.write(1, "saved")

        self.assertEqual(memory.read(1), "saved")
        with self.assertRaises(IndexError):
            memory.read(2)

    def test_initial_data_sets_size_and_round_trips_to_file(self):
        memory = Memory(initialData=[1, "two", True])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            memory.filedump(path)
            restored = Memory()
            restored.fileload(path)

        self.assertEqual(restored.size, 3)
        self.assertEqual(restored.data, [1, "two", True])

    def test_rejects_mismatched_initial_size(self):
        with self.assertRaises(ValueError):
            Memory(3, [1, 2])


class StructureTests(unittest.TestCase):
    def test_vectors(self):
        self.assertEqual(repr(Vector2D(1, 2) + Vector2D(3, 4)), "Vector2D(4, 6)")
        self.assertEqual(Vector2D(1, 2).cross(Vector2D(3, 4)), -2)
        self.assertEqual(repr(Vector3D(1, 0, 0).cross(Vector3D(0, 1, 0))), "Vector3D(0, 0, 1)")
        self.assertEqual(Vector([1, 2, 3]).dot(Vector([4, 5, 6])), 32)
        self.assertEqual(repr(Vector([1, 2])), "Vector([1, 2])")

    def test_matrix_operations(self):
        matrix = Matrix([[4, 7], [2, 6]])

        self.assertEqual(matrix.determinant(), 10)
        self.assertEqual(matrix.matmul(Matrix([[1], [2]])).rows, [[18], [14]])
        inverse = matrix.inverse().rows
        self.assertAlmostEqual(inverse[0][0], 0.6)
        self.assertAlmostEqual(inverse[1][1], 0.4)


class CalculatorTests(unittest.TestCase):
    def setUp(self):
        self.calculator = Calculator()

    def test_integer_sequences_and_factors(self):
        self.assertEqual(self.calculator.fibonacci(0), 0)
        self.assertEqual(self.calculator.fibonacci(10), 55)
        self.assertEqual(self.calculator.prime_factors(84), [2, 2, 3, 7])
        self.assertEqual(self.calculator.prime_factorization(20), {2: 2, 5: 1})

    def test_perfect_power_uses_requested_exponent(self):
        self.assertTrue(self.calculator.is_perfect_power(27, 3))
        self.assertTrue(self.calculator.is_perfect_power(-27, 3))
        self.assertFalse(self.calculator.is_perfect_power(16, 3))
        self.assertFalse(self.calculator.is_perfect_power(-16, 2))

    def test_base_conversion_supports_alphabetic_digits(self):
        self.assertEqual(self.calculator.base_conv("FF", 16, 2), "11111111")
        self.assertEqual(self.calculator.base_conv(35, 10, 36), "Z")

    def test_system_of_equations(self):
        solution = self.calculator.solve_system_of_equations([(1, 1, 3), (2, -1, 0)])

        self.assertEqual(solution.tolist(), [1.0, 2.0])
        with self.assertRaises(ValueError):
            self.calculator.solve_system_of_equations([(1, 2, 3)])

    def test_calculus_helpers(self):
        self.assertAlmostEqual(self.calculator.derivative(lambda value: value**2, 3), 6, places=4)
        self.assertAlmostEqual(self.calculator.integral(lambda value: value, 0, 1), 0.5, places=4)


class ValueTypeTests(unittest.TestCase):
    def test_complex_arithmetic(self):
        left = Complex(2, 3)
        right = Complex(4, -1)

        self.assertEqual(left + right, Complex(6, 2))
        self.assertEqual(left * right, Complex(11, 10))
        self.assertEqual(left.conjugate(), Complex(2, -3))

    def test_time_normalization_and_conversion(self):
        time = Time(1, 59, 90)

        self.assertEqual(str(time), "02:00:30")
        self.assertEqual(Time.from_seconds(3661).to_string(), "01:01:01")
        self.assertEqual((Time(1) + Time(minutes=30)).to_minutes(), 90)


class GUIWrapperTests(unittest.TestCase):
    def test_gui_app_configures_injected_root(self):
        root = mock.Mock()
        root.winfo_width.return_value = 640
        root.winfo_height.return_value = 480
        root.winfo_screenwidth.return_value = 1920
        root.winfo_screenheight.return_value = 1080
        style = mock.Mock()
        style.theme_names.return_value = ("default", "clam")
        content = mock.Mock()

        with mock.patch.object(PythonExtras.ttk, "Style", return_value=style), \
                mock.patch.object(PythonExtras.ttk, "Frame", return_value=content):
            app = GUIApp(
                title="Example",
                size=(640, 480),
                min_size=(320, 240),
                resizable=(True, False),
                theme="clam",
                root=root,
            )
            app.center()

        root.title.assert_called_once_with("Example")
        root.minsize.assert_called_once_with(320, 240)
        root.resizable.assert_called_once_with(True, False)
        style.theme_use.assert_called_once_with("clam")
        root.geometry.assert_called_with("640x480+640+300")

    def test_gui_app_rejects_invalid_sizes(self):
        for size in ((0, 100), (100,), ("100", 100)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                GUIApp._validate_size(size, "size")

    def test_gui_form_collects_and_validates_values(self):
        interpreter = tk.Tcl()
        form = GUIForm.__new__(GUIForm)
        form.fields = {
            "name": {
                "variable": tk.StringVar(interpreter, "Ada"),
                "required": True,
                "validator": None,
            },
            "age": {
                "variable": tk.StringVar(interpreter, "36"),
                "required": True,
                "validator": lambda value: value.isdigit() or "age must be numeric",
            },
            "active": {
                "variable": tk.BooleanVar(interpreter, True),
                "required": False,
                "validator": None,
            },
        }

        self.assertEqual(form.get_values(), {"name": "Ada", "age": "36", "active": True})
        form.set_values(name="Grace", age="invalid")
        with self.assertRaisesRegex(ValueError, "age must be numeric"):
            form.get_values()
        with self.assertRaises(KeyError):
            form.set_values(unknown="value")

    def test_import_does_not_create_tk_root(self):
        self.assertIsNone(tk._default_root)


if __name__ == "__main__":
    unittest.main()