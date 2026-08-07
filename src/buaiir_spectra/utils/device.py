from enum import Enum
from typing import List

class Device(Enum):
    # Device wrapper to avoid not support device errors 
    BIO_SCIENCE = 0
    SCAN_CORDER =  1
    LOW_COST  = 2

    @classmethod
    def get_devices(cls):
        return [member.name for member in cls]

    @classmethod
    def get_devices(cls):
        return list(cls)
    

