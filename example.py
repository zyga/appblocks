#!/usr/bin/env python3
import sys
import logging

from appblocks import block, Block, InputPort, OutputPort, Network
from appblocks.components import PrintText, Constant

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)


@block
def add(a:InputPort(), b:InputPort()) -> OutputPort(primary=True):
    return a + b


"""
class Add(Block):
    a = InputPort()
    b = InputPort()
    result = OutputPort(primary=True)

    def __call__(self):
        self.result = self.a + self.b


class HelloWorldNetwork(Network):
    PrintText(
        text=Add(
            a=Constant("Hello"),
            b=Constant(" World"))


if __name__ == '__main__':
    HelloWorld().run()
"""