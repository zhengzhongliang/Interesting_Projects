class Person:
    def __init__(self, name):
        self.name = name

class Employee(Person):
    def __init__(self,name, position):
        super(Employee, self).__init__(name)   #this is better than the hard coding way of initialization, because the sub class may be inherited from more than one parent class
        self.position = position


thomas = Employee('Thomas','Manager')
thomas.name = "David"

print(thomas.name)
print(thomas.position)