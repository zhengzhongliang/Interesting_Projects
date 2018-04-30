# is it possible to write inheritance class without super function?

class Person:
    def __init__(self, name):
        self.name = name

class Employee(Person):
    def __init__(self,name, position):
        Person.__init__(self, name)
        self.position = position


thomas = Employee('Thomas','Manager')
thomas.name = "David"

print(thomas.name)
print(thomas.position)