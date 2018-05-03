#from abc import ABC, abstractmethod

#class Person():
#  def __init__(self, name):
#    self.name=name
#  #@abstractmethod
#  def _build(self):
#    pass

#class Employee(Person):
#  def __init__(self, name, position):
#    super().__init__(name)
#    self.position=position
#  def _build(self):
#    print(self.position)
#    return 0


#A = Employee('Zhengzhong','GTA')

#A._build()

# the above code can successfully output "GTA". It turns about the override of method is automatic. We do not need to type "@abstractmethod" here. And we do not need to inculde "ABC" in the base class.

#=====================================================


from abc import ABC, abstractmethod

class Person():
  def __init__(self, name):
    self.name=name
  #@abstractmethod
  def _build(self):
    pass

class Employee(Person):
  def __init__(self, name):
    super().__init__(name)

  def _build(self, position):
    self.position=position
    print(self.position)
    return 0


A = Employee('Zhengzhong')

A._build('GTA')

#the code above can run successfully. It shows that the attributes can be added at any functions, not only construction functions.
