REGISTRY = {}

from .basic_controller import BasicMAC
from .basic_controller_ht import BasicMAC_Hetero

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["basic_mac_ht"] = BasicMAC_Hetero