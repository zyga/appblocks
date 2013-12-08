#!/usr/bin/env python3
BROKEN = 1

class Type(type):

    def __new__(mcls, name, bases, namespace):
        if BROKEN:
            for name, value in namespace.items():
                pass
        return type.__new__(mcls, name, bases, namespace)


class Obj(metaclass=Type):

    def __init__(self, **kwargs):
        pass

assert repr(Obj) == "<class '__main__.Obj'>", "was %r" % Obj
