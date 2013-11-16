"""
:module:`appblocks.components` - reusable components for AppBlocks
==================================================================
"""
import sys

from appblocks import Block, OutputPort, InputPort


class Constant(Block):
    """
    A block that wraps a constant value.

    Example::

        class Add(Block):
            a, b = InputPort(), InputPort()
            result = OutputPort()

            def __call__(self):
                self.result = self.a + self.b

        class DoubleNetwork(Network):
            two = Constant(2)
            add = Add(a=two.value, b=two.value)

        assert DoubleNetwork().run().add.result == 4
    """

    value = OutputPort("The constant value", primary=True)

    def __init__(self, value):
        """
        Initialize the constant block with a specified value.
        """
        self._val_value = value


class ReadTextFile(Block):
    """
    A block that reads text from a file
    """

    pathname = InputPort("pathname of the file to read")
    encoding = InputPort("encoding to use", default="UTF-8")
    text = OutputPort("text of the file")
    error = OutputPort("any problem encountered")

    def __call__(self):
        try:
            with open(self.pathname, self.encoding) as stream:
                self.text = stream.read()
        except (OSError, IOError, UnicodeDecodeError) as exc:
            self.error = exc


class PrintText(Block):
    """
    A block that prints text to a stream
    """

    text = InputPort("the text to print")
    stream = InputPort(
        "the stream to print to", default=sys.stdout)

    def __call__(self):
        print(self.text, file=self.stream)


class All(Block):
    input = InputPort()
    result = OutputPort(primary=True)

    def __call__(self):
        self.result = all(input)


class Any(Block):
    input = InputPort()
    result = OutputPort(primary=True)

    def __call__(self):
        self.result = any(input)
