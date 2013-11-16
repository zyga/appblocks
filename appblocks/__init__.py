"""
:module:`appblocks` - command line interface application framework
==================================================================

A new way to write applications, based on the concepts of flow programming.

Introduction
------------

AppBlocks allow you to assemble complex applications out of small, reusable,
testable components. Each component, known as a Block, encapsulates a small
amount of computing, just enough to make sense. Perfect blocks should be well
defined, testable and reusable. Blocks define input and output ports to
communicate with the outside world. Intuitive syntax allows to use ports as if
they were regular variables. Have a look at the hello world example below.

Example::

    class Producer(Block):
        output = OutputPort("the data being produced")

        def __call__(self):
            self.output = "hello world"

    class Consumer(Block):
        input = InputPort("the data being consumed")

        def __call__(self):
            print(self.input)

    class HelloWorld(Network):
        # define and connect blocks
        producer = Producer()
        consumer = Consumer()
        consumer.input = producer.output

    HelloWorld().run()

Break your application into pieces
----------------------------------

The strength of AppBlocks lies in the way it allows developers to create
testable, documented, black boxes that perform a concrete function. You can
think of that as very high-level function. In fact, they very much behave like
regular functions.

Combine applications like never before
--------------------------------------

What makes blocks different is how you assemble them together. Instead of
imperative top-down code that runs one function after another, blocks are
connected together to form networks. Network definitions include analysis
software that ensures everything is connected properly, that no input ports are
forgotten or that no loops are formed.

A network can be executed. AppBlocks will setup a way for each block to execute
once all of its data is ready. This frees the programmer from having to specify
explicit ordering of operations.  It may also speed up certain operations
(though regular python GIL limitations apply).
"""

import unittest
import logging
import concurrent.futures.process


__all__ = [
    'Block',
    'InputPort', 'TriggerPort',
    'OutputPort', 'ErrorPort', 'DonePort', 'ReadyPort',
    'Network',
    'IN', 'OUT',
    'Unset',
]

# Constants for Port.direction
IN = "in"
OUT = "out"

# Logger for appblocks
_logger = logging.getLogger("appblocks")


class UnsetType:
    """
    Special type used for the Unset value.

    This class has only one value, `Unset`. If you access a :class:`Port` from
    inside a :meth:`Block.__call__()` and it was not assigned to before you
    will get this value back.
    """

    _instance = None

    def __new__(cls):
        """
        Instantiate a new UnsetType object (at most one)
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "Unset"


Unset = UnsetType()


class UnsetTests(unittest.TestCase):
    """
    Tests for :class:`Unset`
    """

    def test_repr(self):
        """
        verify how the repr looks like
        """
        self.assertEqual(repr(Unset), "Unset")

    def test_new(self):
        """
        verify that there is really one Unset
        """
        self.assertIs(UnsetType(), UnsetType())

    def test_Unset(self):
        """
        verify that Unset is the instance of UnsetType
        """
        self.assertIs(Unset, UnsetType())


class BoundPort:
    """
    A :class:`Port` instance bound to a :class:`Block` instance.

    You will not be using bound ports directly, they are returned from some of
    the methods of property-like :class:`InputPort` and :class:`OutputPort`
    classes.

    They are used indirectly when forming connections between ports inside
    network definitions. Typically a bound output port instance will be on the
    right hand side of an assignment expression.
    """

    def __init__(self, block, port):
        """
        Initialize a new bound port

        :param block:
            A :class:`Block` object
        :param port:
            An :class:`Port` object
        """
        if not isinstance(block, Block):
            raise TypeError("block must be a Block")
        if not isinstance(port, Port):
            raise TypeError("port must be a Port")
        self.block = block
        self.port = port

    def __repr__(self):
        return "{}({!r}, {!r})".format(
            self.__class__.__name__, self.block, self.port)

    def __str__(self):
        return "port {} bound to {!r}".format(self.port.name, self.block)


class BoundPortTests(unittest.TestCase):
    """
    Tests for :class:`BoundPort`
    """

    def setUp(self):
        self.block = Block()
        self.port = Port()
        self.bound_port = BoundPort(block, port)

    def test_init(self):
        self.assertIs(self.bound_port.block, self.block)
        self.assertIs(self.bound_port.port, self.port)

    def test_repr(self):
        self.assertEqual(repr(self.bound_port), "BoundPort(Blok(), Port())")

    def test_str(self):
        self.assertEqual(str(self.bound_port), "port None bound to Block()")


class BoundInputPort(BoundPort):
    """
    An :class:`InputPort` instance bound to a :class:`Block` instance.
    """

    def __init__(self, block, port):
        """
        Initialize a new bound input port

        :param block:
            A :class:`Block` object
        :param port:
            An :class:`InputPort` object
        """
        if not isinstance(port, InputPort):
            raise TypeError("port must be an InputPort")
        super().__init__(block, port)

    @property
    def connected_to(self):
        """
        The :class:`BoundOutputPort` this BoundInputPort is connected to

        This property is only defined on the input port class as each input is
        connected to at most one output but any output can be connected to any
        number of inputs.
        """
        return getattr(self.block, self.port._con_attr, None)


class BoundOutputPort(BoundPort):
    """
    An :class:`OutputPort` instance bound to a :class:`Block` instance.
    """

    def __init__(self, block, port):
        """
        Initialize a new bound input port

        :param block:
            A :class:`Block` object
        :param port:
            An :class:`OutputPort` object
        """
        if not isinstance(port, OutputPort):
            raise TypeError("port must be a OutputPort")
        super().__init__(block, port)


class Port:
    """
    A input/output abstraction for blocks.

    An output port can be connected to an input port which will cause data to
    flow between blocks. Output blocks may be left disconnected, their values
    will be discarded.

    Accessing Port instances through a Block class simply returns the Port
    object itself.

    Access through Block instances returns either, a BoundPort object, which
    forms a association between a port and a block. Such bound port can be used
    to establish connection to another bound port (or to inspect and break such
    connections).
    """

    def __init__(self, doc=None, name=None):
        """
        Initialize a new port.

        The name argument is optional, all ports created inside a
        :class:`Block` definition will automatically be named properly by the
        block meta-class.
        """
        self.name = name
        self.__doc__ = doc

    def __repr__(self):
        return "{}(name={!r})".format(self.__class__.__name__, self.name)

    def direction(self):
        """
        Direction of the port, either IN or OUT
        """

    @property
    def _value_attr(self):
        """
        Attribute name on Block instances where port value is stored.
        """
        return "_val_{}".format(self.name)

    @property
    def _con_attr(self):
        """
        Attribute name on Block instances where port connection is stored.
        """
        return "_con_{}".format(self.name)


class InputPort(Port):
    """
    A :class:`Port` that allows data to enter a :class:`Block`.

    An output port can be connected to an input port which will cause data to
    flow between blocks. Output ports may be left disconnected.

    Accessing :class:`InputPort` instances through a :class:`Block` class
    simply returns the :class:`InputPort` object itself.

    Access through :class:`Block` instances returns either, a
    :class:`BoundInputPort` object, which forms an association between a port
    and a block. Such bound port can be used to establish connection to another
    bound port (or to inspect and break such connections).
    """

    def __init__(self, doc=None, default=Unset, name=None):
        """
        Initialize a new input port.

        :param doc:
            Documentation of the port instance. Providing a docstring for port
            instances makes it easier for developers to understand the purpose
            of each port.
        :param default:
            A default value which is used if a port is left disconnected.
            Providing a sensible default value for an input port may make it
            easier to use in common use-cases.
        :param name:
            Name (identifier) of the port. This value is usually not provided,
            it is automatically filled in by the metaclass that constructs
            block subclasses.
        """
        super().__init__(doc, name)
        self.default = default

    def __repr__(self):
        return "{}(name={!r}, default={!r})".format(
            self.__class__.__name__, self.name, self.default)      

    @property
    def direction(self):
        """
        Port direction, always ``IN``
        """
        return IN

    def __get__(self, instance, owner):
        """
        The class/instance attribute access descriptor.

        :param instance:
            Instance of the object an :class:`InputPort` is defined within,
            this is most likely a :class:`Block` instance but this is not
            guaranteed.  A value of None is passed when an InputPort is
            accessed through a class rather than through an instance.
        :param owner:
            Class that an InputPort is defined within. This is never None and
            as above, it is most likely a Block subclass.
        :raises AttributeError:
            When accessing a disconnected port from inside
            :meth:`Block.__call__()`.
        :returns:
            * Accessing input ports through classes returns the InputPort
              object directly. This is similar to how properties behave in
              Python.
            * Accessing input ports inside :meth:`Block.__call__()` reads the
              value from the remote end of the port. This may be `Unset` if the
              remote port was not written to yet.
            * Accessing input ports outside of :meth:`Block.__call__()` returns
              a :class:`BoundInputPort` object.

        This descriptor serves multiple purposes. Most importantly it is used
        for accessing input data inside of a block's __call__() method. It is
        also used to connect ports together, inside network definition classes.

        .. note::
            Descriptors are a lesser known internal implementation detail of
            Python. If the description is confusing please familiarize yourself
            with the topic of Python descriptors.
        """
        if instance is None:
            return self
        else:
            if getattr(instance, '_call_in_progress', False):
                bound_output_port = getattr(instance, self._con_attr, None)
                if bound_output_port is None:
                    raise AttributeError("input port is not connected")
                assert isinstance(bound_output_port, BoundOutputPort)
                return getattr(
                    bound_output_port.block,
                    bound_output_port.port._val_attr, Unset)
            else:
                return BoundInputPort(instance, self)

    def __set__(self, instance, value):
        """
        The instance attribute assignment descriptor.

        :param instance:
            Instance of the object an :class:`InputPort` is defined within,
            this is most likely a :class:`Block` instance but this is not
            guaranteed. It is never None.
        :param value:
            A value that is being assigned. This must be a
            :class:`BoundOutputPort` instance or :class:`Block` instance
            that has a :class:``OutputPort` with the :attr:`OutputPort.primary`
            attribute set to True. The second form is used as a shortcut
            for blocks that produce the typical result on a designated output
            port.
        :returns:
            None
        :raises AttributeError:
            When trying to assign any value inside :meth:`Block.__call__()`
        :raises TypeError:
            When trying to assign anything other than :class:`BoundOutputPort`
            outside of :meth:`Block.__call__()`. This is a sign of incorrect
            attempt to create a connection between two ports.
        :raises ValueError:
            When trying to assign a BoundOutputPort that belongs to the same
            block. Loops are not allowed.

        The only use case of assigning to an InputPort is to connect it to a
        BoundOutputPort on another block. This allows developers to use natural
        syntax in their network definitions.

        Example::

            class Net(Network):
                producer = Producer()
                consumer = Consumer()
                # Connect the consumer to the producer
                consumer.input = producer.output

        .. note::
            Descriptors are a lesser known internal implementation detail of
            Python. If the description is confusing please familiarize yourself
            with the topic of Python descriptors.
        """
        if getattr(instance, '_call_in_progress', False):
            raise AttributeError("input ports are read-only")
        else:
            if isinstance(value, Block):
                block = value
                if block._primary_output_port is None:
                    raise ValueError(
                        "block {!r} has no primary output port".format(
                            value.__class__))
                bound_output_port = getattr(block, block._primary_output_port.name)
            elif isinstance(value, BoundOutputPort):
                bound_output_port = value
            else:
                raise TypeError("rvalue must be a BoundOutputPort or a Block")
            if bound_output_port.block is instance:
                raise ValueError("cannot form connections to the same block")
            setattr(instance, self._con_attr, bound_output_port)


class TriggerPort(InputPort):
    """
    A :class:`InputPort` used for explicit delay and sequential triggering

    A trigger port is implicitly added to every :class:`Block`, unless defined
    explicitly. By default all disconnected blocks are auto-triggered. They
    behave is if they were connected to a Constant(True) block. This can be
    overridden both at block definition time and at network definition time.

    A trigger port that is connected to anther port will lose the default value
    and will properly wait for another block to provide the trigger value.

    A trigger port that is defined explicitly in class definition namespace
    prevents the implicit definition from being added. This can be used to
    define trigger points that have no default, must be wired to another port
    and will properly report graph connectivity problem at network definition
    time.

    The choice of trigger port behavior is left to both block and network
    designers.

    Example::

        class Printer(Block):
            text = InputPort()

        class HelloWorld(Network):
            hello = Printer(text=Constant("hello")
            world = Printer(text=Constant(" world")
            world.trigger = hello.done

        HelloWorld().run()
    """


class MultiInputPort(InputPort):
    """
    An :class:`InputPort` that collects data from a number of output ports

    This type of port can be connected to multiple output ports at the same
    time.  It will be ready only after all of the output ports become ready
    though.
    """


class OutputPort(Port):
    """
    A :class:`Port` that allows data to exit a :class:`Block`.

    An output port can be connected to an input port which will cause data to
    flow between blocks. Output blocks may be left disconnected, their values
    will be discarded.

    Accessing :class:`OutputPort` instances through a :class:`Block` class
    simply returns the :class:OutputPort` object itself.

    Access through :class:`Block` instances returns either, a
    :class:`BoundPort` object, which forms a stable association between a port
    and a block. Such bound port can be used to establish connection to another
    bound port (or to inspect and break such connections).

    :ivar name:
        name of the port identifier
    :ivar primary:
        flag indicating that this is the primary (implicit) output port of a
        block.  The primary output port is set to the return value of
        :meth:`Block.__call__()` method. It is also used as the implicit port
        when connecting a port to another block (as opposed to a concrete
        port).
    """

    def __init__(self, doc=None, primary=False, name=None):
        """
        Initialize a new output port.

        The name argument is optional, all ports created inside a
        :class:`Block` definition will automatically be named properly by the
        block meta-class.
        """
        super().__init__(doc, name)
        self.primary = primary

    def __repr__(self):
        return "{}(name={!r}, primary={!r})".format(
            self.__class__.__name__, self.name, self.primary)      

    @property
    def direction(self):
        """
        Port direction, always ``OUT``
        """
        return OUT

    def __get__(self, instance, owner):
        """
        The class/instance attribute access descriptor.

        :param instance:
            Instance of the object an OutputPort is defined within, this is
            most likely a :class:`Block` instance but this is not guaranteed.
            A value of None is passed when a Port is accessed through a class
            rather than through an instance.
        :param owner:
            Class that an OutputPort is defined within. This is never None and
            as above, it is most likely a Block subclass.
        :returns:
            * Accessing output ports through classes returns the OutputPort
              object directly
            * Accessing output ports through instances returns a
              :class:`BoundOutputPort` object that knows about the port and the
              instance it is associated with. When used this way you cannot
              immediately access port "value" but you can use it to setup
              connections between ports in different blocks.
            * Accessing output ports through instances while inside
              :meth:`__call__()` returns the value associated with the port, or
              the :ref:`Unset` object. In this way ports behave just like
              properties or instance attributes.

        This descriptor serves multiple purposes. Most importantly it is used
        for accessing port value inside of block's __call__() method. It is
        also used to connect ports together, inside network definition classes.

        .. note::
            Descriptors are a lesser known internal implementation detail of
            Python. If the description is confusing please familiarize yourself
            with the topic of Python descriptors.
        """
        if instance is None:
            return self
        else:
            if getattr(instance, '_call_in_progress', False):
                return getattr(instance, self._value_attr, Unset)
            else:
                return BoundPort(instance, self)

    def __set__(self, instance, value):
        """
        The instance attribute assignment descriptor.

        :param instance:
            Instance of the object an OutputPort is defined within, this is
            most likely a :class:`Block` instance but this is not guaranteed.
            It is never None.
        :param value:
            A value that is being assigned.
        :returns:
            None
        :raises AttributeError:
            When trying to assign a value outside of :meth:`Block.__call__()`.
            This is meant to catch errors where users incorrectly wire
            something to, rather than from, an output port

        This descriptor allows setting port value for setting port value inside
        of block's __call__() method.

        .. note::
            Descriptors are a lesser known internal implementation detail of
            Python. If the description is confusing please familiarize yourself
            with the topic of Python descriptors.
        """
        if getattr(instance, '_call_in_progress', False):
            setattr(instance, self._value_attr, value)
        else:
            raise AttributeError("cannot form connections *to* output ports")


class DonePort(OutputPort):
    """
    A :class:`OutputPort` used to signal that a block is done executing.

    A done port is implicitly added to every :class:`Block`, unless defined
    explicitly. It is also automatically set by the code that invokes
    :meth:`Block.__call__()`.

    It is designed to be wired to a :class:`TriggerPort` for explicit
    sequential control over execution control flow.
    """


class ErrorPort(OutputPort):
    """
    A :class:`OutputPort` used to signal block execution errors.

    An error port is implicitly added to every :class:`Block`, unless defined
    explicitly. It is also automatically set by the code that invokes
    :meth:`Block.__call__()` when an exception escapes from `__call__()`. When
    a block executes without problems it is not set to anything and won't
    activate the blocks that are connected to it.

    It is designed to be connected to other blocks that can do error handling.
    Since it is always present on every block it allows network designers to
    account for error handling even if any particular block is not doing that
    by itself.
    """


class ReadyPort(OutputPort):
    """
    A :class:`OutputPort` used to signal that a block is ready for execution.

    A ready port is implicitly added to every :class:`Block`, unless defined
    explicitly. It is also automatically set to True by the code that invokes
    :meth:`Block.__call__()` just prior to the actual execution.

    It is designed to be connected to code that needs to start at the same time
    another block is started, without having to understand the precise set of
    conditions.
    """


class BlockType(type):
    """
    Type of all block classes.

    The BlockType class is responsible for collecting all of the port objects
    into a _ports dictionary. It supports inheritance so ports from all of the
    base classes are collected.
    """

    def __new__(mcls, name, bases, namespace):
        """
        Construct a new Block class.

        This collects all of the :class:`Port` instances from this and base
        classes and assigns that to a `_ports` attribute on the new class.
        """
        # Inject special ports, unless already defined
        if 'trigger' not in namespace:
            namespace['trigger'] = TriggerPort(
                doc='input port for control over execution flow (implicit)',
                default=True)
        if 'ready' not in namespace:
            namespace['ready'] = ReadyPort(
                doc='output port set when block becomes ready (implicit)')
        if 'done' not in namespace:
            namespace['done'] = DonePort(
                doc='output port set when block is done executing (implicit)')
        if 'error' not in namespace:
            namespace['error'] = ErrorPort(
                doc='output port set to escaped exceptions (implicit)')
        # Compute and store the _ports class-attribute
        ports = {}
        for base in bases:
            if issubclass(base, Block):
                ports.update(base._ports)
        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, Port):
                ports[attr_name] = attr_value
        namespace['_ports'] = ports
        # Set name of any port that needs it
        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, Port) and attr_value.name is None:
                attr_value.name = attr_name
        # Compute and store the _primary_output_port class-attribute
        primary_list = [
            port for port in ports.values() 
            if port.direction == OUT and port.primary]
        if len(primary_list) > 1:
            raise ValueError("only one port per block may be primary")
        namespace['_primary_output_port'] = (
            primary_list[0] if len(primary_list) == 1
            else None)
        _logger.debug("Creating block %r with ports %r", name, ports)
        return type.__new__(mcls, name, bases, namespace)


class Block(metaclass=BlockType):
    """
    A piece of application logic.

    Example usage::

        class Filter(Block):
            '''
            A really simple filter component
            '''

            input = InputPort()
            output = OutputPort()

            def __call__(self):
                self.output = self.input

    A block can only communicate with the outside world by connecting ports
    together. Ports bring data in and out of a block.

    :ivar _call_in_progress:
        Flag indicating that all ports should trigger access to their
        corresponding data value. This is automatically set when
        :meth:`__call__()` is called. It is deleted at other times.

    :cvar _ports:
        A dictionary with all of the :class:`Port` objects.
    """

    def __init__(self, **kwargs):
        """
        Initialize a new block.

        All keyword arguments are interpreted as port assignments. This can
        simplify connecting multiple blocks together as in many cases temporary
        variables won't need to be used.
        """
        for port_name, port_value in kwargs.items():
            if port_name not in self._ports:
                raise AttributeError(
                    "there is no port called {}".format(port_name))
            setattr(self, port_name, port_value)

    def __repr__(self):
        # TODO: display ports maybe?
        return "<{} object at {:x}>".format(self.__class__.__name__, id(self))

    def __call__(self):
        """
        Abstract method that performs operations encapsulated by this block.

        This method must be overridden by subclasses. The return value is
        irrelevant and is discarded. It is expected that all of the output
        ports are assigned by the code inside (otherwise they will remain
        unset).
        """

    def _is_ready(self):
        """
        Check if this block is ready to be executed.
        """
        for port in self._ports:
            if isinstance(port, InputPort):
                if getattr(self, port.name) is Unset:
                    return False
        return True

    def _execute(self):
        """
        Execute the code encapsulated by this block.

        This method should not be called directly. It is called by
        :class:`Network` when all of the ports are wired and ready and a free
        execution unit is available.
        """
        self._call_in_progress = True
        self.ready = True
        try:
            retval = self()
            if self._primary_output_port:
                setattr(self, self._primary_output_port.name, retval)
        except Exception as exc:
            self.error = exc
        else:
            self.done = True
        finally:
            del self._call_in_progress


class BlockTests(unittest.TestCase):
    """
    Tests for :class:`Block`
    """

    def setUp(self):
        class Producer(Block):
            output = OutputPort()

            def __call__(self):
                self.output = "data"

        class Consumer(Block):
            input = InputPort()

            def __call__(self):
                print(self.input)

        self.Producer = Producer
        self.Consumer = Consumer

    def test_class_level_port_access(self):
        """
        verify that accessing ports from a class returns the port itself
        """
        self.assertIsInstance(self.Producer.output, OutputPort)
        self.assertIsInstance(self.Consumer.input, InputPort)

    def test_instance_level_port_access(self):
        """
        verify that accessing ports from an instance returns a BoundPort
        """
        self.assertIsInstance(self.Producer().output, BoundOutputPort)
        self.assertIsInstance(self.Consumer().input, BoundInputPort)

    def test_bound_port_properties(self):
        """
        verify that bound ports are initialized correctly
        """
        producer = self.Producer()
        self.assertIs(producer.output.block, producer)
        self.assertIs(producer.output.port, self.Producer.output)
        consumer = self.Consumer()
        self.assertIs(consumer.input.block, consumer)
        self.assertIs(consumer.input.block, self.Consumer.input)

    def test_wiring_blocks(self):
        """
        verify that basic producer-consumer network can be wired together
        """
        class TestNetwork(Network):
            producer = self.Producer()
            consumer = self.Consumer()
            consumer.input = producer.output

        net = TestNetwork()
        self.assertIs(net.consumer.input.connected_to, net.producer.output)


class NetworkType(type):
    """
    Type of all network classes.

    The NetworkType class is responsible for collecting all of the block objects
    into a _blocks set.
    """

    def __new__(mcls, name, bases, namespace):
        blocks = set()
        if False:
            # XXX: disabled because of unclear inheritance semantics
            for base in bases:
                if isinstance(base, Network):
                    blocks.update(base._blocks)
        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, Block):
                blocks.add(attr_value)
        for block in blocks:
            for port in block._ports:
                if port.direction == IN:
                    bound_input_port = getattr(block, port.name)
                    bound_output_port = bound_input_port.connected_to
                    if bound_output_port is not None:
                        other_block = bound_output_port.block
                        blocks.add(other_block)
        namespace['_blocks'] = blocks
        _logger.debug("Creating network %r with blocks %r", name, blocks)
        cls = type.__new__(mcls, name, bases, namespace)
        if bases:
            cls._verify_topology()
        return cls


class TopologyError(Exception):
    """
    Exception raised if there is a problem with the way a network is wired.
    """


class DeadLockError(Exception):
    """
    Exception raised if a network dead-lock is detected.

    A dead-lock occurs when a there are blocks left unprocessed but
    no further blocks can be processed.
    """


class Network(metaclass=NetworkType):
    """
    A canvas to connect blocks together

    Example::

        class Producer(Block):
            output = OutputPort()

        class Consumer(Block):
            input = InputPort()

        class TestNetwork(Network):
            producer = Producer()
            consumer = Consumer()
            consumer.input = producer.output
    """

    # TODO: use inheritance to support a form of network code reuse
    # TODO: do something smart inside instantiated network, maybe keep all of
    # the data flow inside there?
    # TODO: have a method to execute the network in an executor (probably
    # future based)

    @classmethod
    def _verify_topology(cls):
        if not cls._blocks:
            raise TopologyError("no blocks at all")
        initial_list = [
            block for block in cls._blocks
            if block._is_ready]
        if not initial_list:
            raise TopologyError("no blocks are initially ready")
        # TODO: detect loops
        # TODO: detect disconnected error output ports
        # TODO: detect disconnected input ports

    def run(self, executor=None):
        """
        Execute the network.

        :param executor:
            :class:`concurrent.futures.Executor` object to use.
        :returns:
            The run method always returns self, so that simple
            computations can refer to network blocks on one line.

        If the executor object is not provided explicitly a default Thread
        executor is constructed and used. It won't offer any parallel execution
        because of python GIL (global interpreter lock) but should be a safe
        default.

        When designed properly a process executor should offer improved
        performance as all ready blocks are scheduled for execution ion a
        separate process.
        """
        if executor is None:
            executor = concurrent.futures.process.ProcessPoolExecutor()
        try:
            todo = self._blocks.copy()
            while todo:
                ready = [block for block in todo if block._is_ready]
                if not ready:
                    raise DeadLockError("dead-lock with: {!r}".format(todo))
                for block in ready:
                    result = executor.submit(block._execute)
        finally:
            executor.shutdown()