from enum import Enum
from typing import List

class Device(Enum):
    """
    Provides the objects of the 3 devices used during spectra data collection

    Devices:
        BIO_SCIENCE -> High End spectrometer
        LOW_COST -> In house built low cost device
        SCAN_CODER -> Mid Range Reflectance Spectrometer
    """
    BIO_SCIENCE = 0
    SCAN_CODER = 1
    LOW_COST = 2

    @classmethod
    def get_devices(cls):
        """Returns the names of the devices in the enum"""
        return list(cls)
    
    @classmethod
    def get_device_names(cls) -> List[str]:
        """
        Returns the names of the devices used i the experiment
        """
        return [device.name for device in cls]