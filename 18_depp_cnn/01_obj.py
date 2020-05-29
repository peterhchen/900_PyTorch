class Lizard:
    def __init__(self, name):
        self.name = name
    
    def set_name(self, name):
        self.name = name
    
lizard = Lizard ('deep')
print (lizard.name)

lizard.set_name('lizard')
print (lizard.name)
