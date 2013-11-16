#!/usr/bin/env python3
import sys
import logging

from appblocks import Block, InputPort, OutputPort, Network
from appblocks.components import PrintText, Constant

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)


class Add(Block):
    a = InputPort()
    b = InputPort()
    result = OutputPort(primary=True)

    def __call__(self):
        self.result = self.a + self.b


class HelloWorldNetwork(Network):
    printer = PrintText()
    printer.text = Add(a=Constant("Hello"), b=Constant(" World"))


if __name__ == '__main__':
    HelloWorld().run()
