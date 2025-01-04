REGISTRY = {}

from .rnn_agent import RNNAgent
from .fq_worker_agent import FeUdalWorker
from .fq_manager_agent import FeUdalManager

REGISTRY["rnn"] = RNNAgent

REGISTRY["feudal_worker"] = FeUdalWorker
REGISTRY["feudal_manager"] = FeUdalManager