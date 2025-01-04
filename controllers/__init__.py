from .basic_controller import BasicMAC

from .fq_controller import FeUdalMAC # 導入FeUdalMAC

REGISTRY = dict()

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["feudal_mac"] = FeUdalMAC # 註冊feudal_mac

