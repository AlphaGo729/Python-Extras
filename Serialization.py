"""Utilities for serializing sequences of strings with escaping."""


def _validate_characters(delim, escape):
    if not isinstance(delim, str) or len(delim) != 1:
        raise ValueError("Delimiter must be a single character")
    if not isinstance(escape, str) or len(escape) != 1:
        raise ValueError("Escape must be a single character")
    if delim == escape:
        raise ValueError("Delimiter and escape characters must be different")


def Encode(values, delim="|", escape="\\"):
    """Encode an iterable of values into a delimiter-separated string."""
    _validate_characters(delim, escape)
    encoded_values = []
    for value in values:
        text = str(value)
        text = text.replace(escape, escape + escape)
        text = text.replace(delim, escape + delim)
        encoded_values.append(text + delim)
    return "".join(encoded_values)


def Decode(text, delim="|", escape="\\"):
    """Decode a string produced by :func:`Encode`."""
    _validate_characters(delim, escape)
    if not isinstance(text, str):
        raise TypeError("Encoded data must be a string")
    if not text:
        return []

    values = []
    current = []
    escaped = False
    for character in text:
        if escaped:
            current.append(character)
            escaped = False
        elif character == escape:
            escaped = True
        elif character == delim:
            values.append("".join(current))
            current = []
        else:
            current.append(character)

    if escaped:
        raise ValueError("Encoded data ends with an incomplete escape sequence")
    if current:
        raise ValueError("Encoded data is missing its final delimiter")
    return values


encode = Encode
decode = Decode


