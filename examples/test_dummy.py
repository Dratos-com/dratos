def add(a, b):
    """
    Adds two numbers together.

    >>> add(2, 3)
    5
    >>> add(-1, 1)
    0
    >>> add(0, 0)
    0
    """
    return a + b

def reverse_string(s):
    """
    Reverses a given string.

    >>> reverse_string("hello")
    'olleh'
    >>> reverse_string("world")
    'dlrow'
    >>> reverse_string("")
    ''
    """
    return s[::-1]
