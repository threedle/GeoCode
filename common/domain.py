from enum import Enum

class Domain(Enum):
    chair = 'chair'
    vase = 'vase'
    table = 'table'

    def __str__(self):
        return self.value
