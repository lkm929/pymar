from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner

from .fq_learner import FeUdalQLearner  # 導入FeUdalQLearner
from .gpq_learner import GPQLearner  # 導入GPQLearner

REGISTRY = dict()

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner

REGISTRY["feudal_qlearner"] = FeUdalQLearner  # 註冊feudal_qlearner
REGISTRY["gp_qlearner"] = GPQLearner  # 註冊gp_qlearner