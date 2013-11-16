AppBlocks
=========

Flow based programming framework for python command line applications.

Why?
====

For better control, testing and reliability of command line apps.

Visual Blocks
=============

```python
class ProductPage(Page):

    def render(self):
        product = self.args.product
        print("[Product {}]".format(product))
        print(textwrap.wrap(product.description))
        print("-" * 78)
        print("Price: {}".format(product.price))
```

FlowBlocks
==========


AppBlocks allows you to create command line applications out of interconnected
elements. Each element performs a dedicated function such as computation,
presentation or user interaction.

Each element class is easy to test with standard best practices. The flow of
application elements is easy to visualise through static analysis or runtime
annotations.
