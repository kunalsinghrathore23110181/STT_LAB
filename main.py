class Calc:
    def __init__(self):
        self.a, self.b = 12, 7

    def add(self):
        return self.a + self.b

    def sub(self):
        return self.a - self.b

    def mul(self):
        return self.a * self.b

    def div(self):
        return self.a / self.b

    def run(self):
        print(self.add())
        print(self.sub())
        print(self.mul())
        print(self.div())


if __name__ == "__main__":
    obj = Calc()
    obj.run()

