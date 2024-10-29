from enum import Enum

class Domain(Enum):
    chair = 'chair'
    vase = 'vase'
    table = 'table'
    ceiling_lamp = 'ceiling_lamp'

    def __str__(self):
        return self.value
