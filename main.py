"""This module performs basic arithmetic calculations using a class."""


class Calc:
    """A simple calculator class to perform basic operations."""

    def __init__(self):
        """Initialize with default values."""
        self.a, self.b = 12, 7

    def add(self):
        """Return the sum of a and b."""
        return self.a + self.b

    def sub(self):
        """Return the difference of a and b."""
        return self.a - self.b

    def mul(self):
        """Return the product of a and b."""
        return self.a * self.b

    def div(self):
        """Return the division of a by b."""
        return self.a / self.b

    def run(self):
        """Run all operations and print results."""
        print(self.add())
        print(self.sub())
        print(self.mul())
        print(self.div())


if __name__ == "__main__":
    Calc().run()
