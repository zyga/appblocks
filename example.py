#!/usr/bin/env python3
import sys

from appblocks import Block, InputPort, OutputPort, Network
from appblocks.components import PrintText, Constant

class Add(Block):
    a = InputPort()
    b = InputPort()
    result = OutputPort(primary=True)

    def __call__(self):
        self.result = self.a + self.b


class HelloWorld(Network):
    print = PrintText()
    print.text = Add(a=Constant("Hello"), b=Constant(" World"))


if __name__ == '__main__':
    HelloWorld().run()
