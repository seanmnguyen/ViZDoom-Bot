# -----------------------------------------------------------------------------
# Agent Imports
# -----------------------------------------------------------------------------

from q_late_fusion import DQNAgent as DQNAgent_LateFusionGrayCenter
from q_late_fusion_rgb import DQNAgent as DQNAgent_LateFusionRGBCenter
from q_cnn import DQNAgent as DQNAgent_CNN
from q_cnn_rgb import DQNAgent as DQNAgent_CNNRGB
from q_rainbow_rgb import DQNAgent as DQNAgent_RainbowRGB
from q_rainbow_stacked_gray_center import DQNAgent as DQNAgent_RainbowLazyStackGrayCenter
from q_rainbow_stacked_gray_center import FrameStack as DQNFrameStackRainbowGrayCenter
from q_rainbow_stacked_gray_center import FRAME_STACK_SIZE as DQN_FRAME_STACK_SIZE_RAINBOW_GRAY_CENTER
from q_rainbow_stacked_rgb_center import DQNAgent as DQNAgent_RainbowLazyStackRGBCenter
from q_rainbow_stacked_rgb_center import FrameStack as DQNFrameStackRainbowRGBCenter
from q_rainbow_stacked_rgb_center import FRAME_STACK_SIZE as DQN_FRAME_STACK_SIZE_RAINBOW_RGB_CENTER
from q_rainbow_stacked_gray_corridor import DQNAgent as DQNAgent_RainbowLazyStackGrayCorridor
from q_rainbow_stacked_gray_corridor import FrameStack as DQNFrameStackRainbowGrayCorridor
from q_rainbow_stacked_gray_corridor import FRAME_STACK_SIZE as DQN_FRAME_STACK_SIZE_RAINBOW_GRAY_CORRIDOR
import q_rainbow_stacked_gray_center as rainbow_lazy_mod_gray_center
import q_rainbow_stacked_rgb_center as rainbow_lazy_mod_rgb_center
import q_rainbow_stacked_gray_corridor as rainbow_lazy_mod_gray_corridor
from ppo_cnn import PPOAgent
# from ppo_cnn_rgb import PPOAgent as PPOAgent_CNNRGBLine
from ppo_cnn_gray import PPOAgent as PPOAgent_Gray
from ppo_cnn_gray import FrameStack as PPOFrameStackGray
from ppo_cnn_gray import FRAME_STACK_SIZE as PPO_FRAME_STACK_SIZE_CNN_GRAY
from ppo_cnn_gray_corridor import PPOAgent as PPOAgent_CNNGrayCorridor
from ppo_cnn_gray_deathmatch import PPOAgent as PPOAgent_CNNGrayDeathmatch
from ppo_cnn_gray_deathmatch import FrameStack as PPOFrameStack_CNNGrayDeathmatch
from ppo_cnn_gray_deathmatch import FRAME_STACK_SIZE as PPO_FRAME_STACK_SIZE_CNN_GRAY_DEATHMATCH
from ppo_cnn_stacked_gray_corridor import PPOAgent as PPOAgent_CNNStackedGrayCorridor
from ppo_cnn_stacked_gray_corridor import FrameStack as PPOFrameStack_CNNGrayCorridor
from ppo_cnn_stacked_gray_corridor import FRAME_STACK_SIZE as PPO_FRAME_STACK_SIZE_CNN_GRAY_CORRIDOR
from ppo_late_fusion_gray_line import PPOAgent as PPOAgent_LateFusionGrayLine
from ppo_late_fusion_rgb_line import PPOAgent as PPOAgent_LateFusionRGBLine
from ppo_cnn_rgb_center import PPOAgent as PPOAgent_CNNRGBCenter
from ppo_late_fusion_rgb_center import PPOAgent as PPOAgent_LateFusionRGBCenter
from ppo_late_fusion_rgb import PPOAgent as PPOAgent_LateFusionRGB
from ppo_late_fusion_rgb import FrameStackRGB as PPOFrameStackRGB
from ppo_late_fusion_rgb import FRAME_STACK_SIZE as PPO_FRAME_STACK_SIZE_LATE_FUSION_RGB
from ppo_late_fusion_rgb_corridor import PPOAgent as PPOAgent_LateFusionRGBCorridor
from ppo_late_fusion_rgb_corridor import FrameStackRGB as PPOFrameStackRGBCorridor
from ppo_late_fusion_rgb_corridor import FRAME_STACK_SIZE as PPO_FRAME_STACK_SIZE_LATE_FUSION_RGB_CORRIDOR
from ppo_late_fusion_gray import PPOAgent as PPOAgent_LateFusionGray
from ppo_late_fusion_gray import FrameStack as PPOFrameStack_LateFusionGray
from ppo_late_fusion_gray import FRAME_STACK_SIZE as PPO_FRAME_STACK_SIZE_LATE_FUSION_GRAY
from ppo_late_fusion_gray_corridor import PPOAgent as PPOAgent_LateFusionGrayCorridor
from ppo_late_fusion_gray_corridor import FrameStack as PPOFrameStack_LateFusionGrayCorridor
from ppo_late_fusion_gray_corridor import FRAME_STACK_SIZE as PPO_FRAME_STACK_SIZE_LATE_FUSION_GRAY_CORRIDOR
from ppo_late_fusion_gray_deathmatch import PPOAgent as PPOAgent_LateFusionGrayDeathmatch
from ppo_late_fusion_gray_deathmatch import FrameStack as PPOFrameStack_LateFusionGrayDeathmatch
from ppo_late_fusion_gray_deathmatch import FRAME_STACK_SIZE as PPO_FRAME_STACK_SIZE_LATE_FUSION_GRAY_DEATHMATCH
from ppo_film_gray_center import PPOAgent as PPOAgent_FiLMGrayCenter
from ppo_film_gray_center import FrameStack as PPOFrameStack_FiLMGrayCenter
from ppo_film_gray_center import FRAME_STACK_SIZE as PPO_FRAME_STACK_SIZE_FILM_GRAY_CENTER
from ppo_film_gray_corridor import PPOAgent as PPOAgent_FiLMGrayCorridor
from ppo_film_gray_corridor import FrameStack as PPOFrameStack_FiLMGrayCorridor
from ppo_film_gray_corridor import FRAME_STACK_SIZE as PPO_FRAME_STACK_SIZE_FILM_GRAY_CORRIDOR
from ppo_film_gray_deathmatch import PPOAgent as PPOAgent_FiLMGrayDeathmatch
from ppo_film_gray_deathmatch import FrameStack as PPOFrameStack_FiLMGrayDeathmatch
from ppo_film_gray_deathmatch import FRAME_STACK_SIZE as PPO_FRAME_STACK_SIZE_FILM_GRAY_DEATHMATCH
from ppo_film_factorized_gray import PPOAgent as PPOAgent_FiLMFactorized
from ppo_film_factorized_gray import FrameStack as PPOFrameStack_FiLMFactorized
from ppo_film_factorized_gray import FRAME_STACK_SIZE as PPO_FRAME_STACK_SIZE_FILM_FACTORIZED
from ppo_film_factorized_gray import FactorizedActionMapper
from ppo_film_factorized_gray_cig import PPOAgent as PPOAgent_FiLMFactorizedCig
from ppo_film_factorized_gray_cig import FrameStack as PPOFrameStack_FiLMFactorizedCig
from ppo_film_factorized_gray_cig import FRAME_STACK_SIZE as PPO_FRAME_STACK_SIZE_FILM_FACTORIZED_CIG
from ppo_film_factorized_gray_cig import FactorizedActionMapper as FactorizedActionMapperCig

# -----------------------------------------------------------------------------
# Model registry
# -----------------------------------------------------------------------------

MODEL_DEFAULT_SCENARIO = {
    "q_cnn": "defend_the_line.cfg",
    "q_cnn_rgb": "defend_the_line.cfg",
    "q_late_fusion_gray_center": "defend_the_center.cfg",
    "q_late_fusion_rgb": "defend_the_center.cfg",
    "ppo_cnn_gray_line": "defend_the_line.cfg",
    "ppo_cnn_rgb_line": "defend_the_line.cfg",
    "ppo_cnn_gray": "defend_the_center.cfg",
    "ppo_cnn_rgb_center": "defend_the_center.cfg",
    "ppo_cnn_gray_corridor": "deadly_corridor.cfg",
    "ppo_cnn_stacked_gray_corridor": "deadly_corridor.cfg",
    "ppo_cnn_stacked_gray_deathmatch": "deathmatch.cfg",
    "ppo_late_fusion_gray_line": "defend_the_line.cfg",
    "ppo_late_fusion_rgb_line": "defend_the_line.cfg",
    "ppo_late_fusion_rgb_center": "defend_the_center.cfg",
    "q_late_fusion_rgb_DC": "deadly_corridor.cfg",
    "q_rainbow_rgb": "defend_the_center.cfg",
    "q_rainbow_stacked_rgb_center": "defend_the_center.cfg",
    "q_rainbow_stacked_gray_center": "defend_the_center.cfg",
    "q_rainbow_stacked_gray_corridor": "defend_the_center.cfg",
    "ppo_cnn_deadly_corridor_gray": "deadly_corridor.cfg",
    "ppo_late_fusion_rgb": "defend_the_center.cfg",
    "ppo_late_fusion_rgb_corridor": "deadly_corridor.cfg",
    "ppo_late_fusion_gray": "defend_the_center.cfg",
    "ppo_late_fusion_gray_corridor": "deadly_corridor.cfg",
    "ppo_late_fusion_gray_deathmatch": "deathmatch.cfg",
    "ppo_film_gray_center": "defend_the_center.cfg",
    "ppo_film_gray_corridor": "deadly_corridor.cfg",
    "ppo_film_gray_deathmatch": "deathmatch.cfg",
    "ppo_film_factorized_gray": "deathmatch.cfg",
    "ppo_film_factorized_gray_cig": "cig_learning.cfg",
}

AGENT_BY_MODEL = {
    "q_cnn": DQNAgent_CNN,
    "q_cnn_rgb": DQNAgent_CNNRGB,
    "q_late_fusion_gray_center": DQNAgent_LateFusionGrayCenter,
    "q_late_fusion_rgb": DQNAgent_LateFusionRGBCenter,
    "ppo_cnn_gray_line": PPOAgent,
    "ppo_cnn_gray": PPOAgent_Gray,
    "ppo_cnn_rgb_center": PPOAgent_CNNRGBCenter,
    "ppo_cnn_gray_corridor": PPOAgent_CNNGrayCorridor,
    "ppo_cnn_stacked_gray_corridor": PPOAgent_CNNStackedGrayCorridor,
    "ppo_cnn_stacked_gray_deathmatch": PPOAgent_CNNGrayDeathmatch,
    "ppo_late_fusion_gray_line": PPOAgent_LateFusionGrayLine,
    "ppo_late_fusion_rgb_line": PPOAgent_LateFusionRGBLine,
    "ppo_late_fusion_rgb_center": PPOAgent_LateFusionRGBCenter,
    "q_late_fusion_rgb_DC": DQNAgent_LateFusionRGBCenter,
    "q_rainbow_rgb": DQNAgent_RainbowRGB,
    "q_rainbow_stacked_rgb_center": DQNAgent_RainbowLazyStackRGBCenter,
    "q_rainbow_stacked_gray_center": DQNAgent_RainbowLazyStackGrayCenter,
    "q_rainbow_stacked_gray_corridor": DQNAgent_RainbowLazyStackGrayCorridor,
    "ppo_late_fusion_rgb": PPOAgent_LateFusionRGB,
    "ppo_late_fusion_rgb_corridor": PPOAgent_LateFusionRGBCorridor,
    "ppo_late_fusion_gray": PPOAgent_LateFusionGray,
    "ppo_late_fusion_gray_corridor": PPOAgent_LateFusionGrayCorridor,
    "ppo_late_fusion_gray_deathmatch": PPOAgent_LateFusionGrayDeathmatch,
    "ppo_film_gray_center": PPOAgent_FiLMGrayCenter,
    "ppo_film_gray_corridor": PPOAgent_FiLMGrayCorridor,
    "ppo_film_gray_deathmatch": PPOAgent_FiLMGrayDeathmatch,
    "ppo_film_factorized_gray": PPOAgent_FiLMFactorized,
    "ppo_film_factorized_gray_cig": PPOAgent_FiLMFactorizedCig,
}

RESOLUTION_BY_MODEL = {
    "q_cnn": (30, 45),
    "q_cnn_rgb": (96, 128),
    "q_late_fusion_gray_center": (96, 128),
    "q_late_fusion_rgb": (96, 128),
    "ppo_cnn_gray_line": (30, 45),
    "ppo_cnn_rgb_line": (30, 45),
    "ppo_cnn_gray": (96, 128),
    "ppo_cnn_rgb_center": (30, 45),
    "ppo_cnn_gray_corridor": (30, 45),
    "ppo_cnn_stacked_gray_corridor": (96, 128),
    "ppo_cnn_stacked_gray_deathmatch": (96, 128),
    "ppo_late_fusion_gray_line": (30, 45),
    "ppo_late_fusion_rgb_line": (30, 45),
    "ppo_late_fusion_rgb_center": (30, 45),
    "q_late_fusion_rgb_DC": (96, 128),
    "q_rainbow_rgb": (96, 128),
    "q_rainbow_stacked_rgb_center": (96, 128),
    "q_rainbow_stacked_gray_center": (96, 128),
    "q_rainbow_stacked_gray_corridor": (96, 128),
    "ppo_cnn_deadly_corridor_gray": (30, 45),
    "ppo_late_fusion_rgb": (96, 128),
    "ppo_late_fusion_rgb_corridor": (96, 128),
    "ppo_late_fusion_gray": (96, 128),
    "ppo_late_fusion_gray_corridor": (96, 128),
    "ppo_late_fusion_gray_deathmatch": (96, 128),
    "ppo_film_gray_center": (96, 128),
    "ppo_film_gray_corridor": (96, 128),
    "ppo_film_gray_deathmatch": (96, 128),
    "ppo_film_factorized_gray": (96, 128),
    "ppo_film_factorized_gray_cig": (96, 128),
}

GRAYSCALE = "GRAY8"
RGB = "RGB24"
AUTO = "AUTO"  # Deterimine from model attributes
COLOR_BY_MODEL = {
    "q_cnn": GRAYSCALE,
    "q_cnn_rgb": RGB,
    "q_late_fusion_gray_center": GRAYSCALE,
    "q_late_fusion_rgb": RGB,
    "ppo_cnn_gray_line": GRAYSCALE,
    "ppo_cnn_rgb_line": RGB,
    "ppo_cnn_gray": GRAYSCALE,
    "ppo_cnn_rgb_center": RGB,
    "ppo_cnn_gray_corridor": GRAYSCALE,
    "ppo_cnn_stacked_gray_corridor": GRAYSCALE,
    "ppo_cnn_stacked_gray_deathmatch": GRAYSCALE,
    "ppo_late_fusion_gray_line": GRAYSCALE,
    "ppo_late_fusion_rgb_line": RGB,
    "ppo_late_fusion_rgb_center": RGB,
    "q_late_fusion_rgb_DC": RGB,
    "q_rainbow_rgb": RGB,
    "q_rainbow_stacked_rgb_center": RGB,
    "q_rainbow_stacked_gray_center": GRAYSCALE,
    "q_rainbow_stacked_gray_corridor": GRAYSCALE,
    "ppo_cnn_deadly_corridor_gray": GRAYSCALE,
    "ppo_late_fusion_rgb": RGB,
    "ppo_late_fusion_rgb_corridor": RGB,
    "ppo_late_fusion_gray": GRAYSCALE,
    "ppo_late_fusion_gray_corridor": GRAYSCALE,
    "ppo_late_fusion_gray_deathmatch": GRAYSCALE,
    "ppo_film_gray_center": GRAYSCALE,
    "ppo_film_gray_corridor": GRAYSCALE,
    "ppo_film_gray_deathmatch": GRAYSCALE,
    "ppo_film_factorized_gray": GRAYSCALE,
    "ppo_film_factorized_gray_cig": GRAYSCALE,
}

PPO_MODELS = {
    "ppo_cnn_gray_line",
    "ppo_cnn_rgb_line",
    "ppo_cnn_gray",
    "ppo_cnn_rgb_center",
    "ppo_cnn_gray_corridor",
    "ppo_cnn_stacked_gray_corridor",
    "ppo_cnn_stacked_gray_deathmatch",
    "ppo_late_fusion_gray_line",
    "ppo_late_fusion_rgb_line",
    "ppo_late_fusion_rgb_center",
    "ppo_cnn_deadly_corridor_gray",
    "ppo_late_fusion_rgb",
    "ppo_late_fusion_rgb_corridor",
    "ppo_late_fusion_gray",
    "ppo_late_fusion_gray_corridor",
    "ppo_late_fusion_gray_deathmatch",
    "ppo_film_gray_center",
    "ppo_film_gray_corridor",
    "ppo_film_gray_deathmatch",
    "ppo_film_factorized_gray",
    "ppo_film_factorized_gray_cig",
}

# Maps models to FrameStack class
FRAME_STACK_MODELS = {
    "q_rainbow_stacked_rgb_center": DQNFrameStackRainbowRGBCenter,
    "q_rainbow_stacked_gray_center": DQNFrameStackRainbowGrayCenter,
    "q_rainbow_stacked_gray_corridor": DQNFrameStackRainbowGrayCorridor,
    "ppo_cnn_gray": PPOFrameStackGray,
    "ppo_cnn_stacked_gray_corridor": PPOFrameStack_CNNGrayCorridor,
    "ppo_cnn_stacked_gray_deathmatch": PPOFrameStack_CNNGrayDeathmatch,
    "ppo_late_fusion_rgb": PPOFrameStackRGB,
    "ppo_late_fusion_rgb_corridor": PPOFrameStackRGBCorridor,
    "ppo_late_fusion_gray": PPOFrameStack_LateFusionGray,
    "ppo_late_fusion_gray_corridor": PPOFrameStack_LateFusionGrayCorridor,
    "ppo_late_fusion_gray_deathmatch": PPOFrameStack_LateFusionGrayDeathmatch,
    "ppo_film_gray_center": PPOFrameStack_FiLMGrayCenter,
    "ppo_film_gray_corridor": PPOFrameStack_FiLMGrayCorridor,
    "ppo_film_gray_deathmatch": PPOFrameStack_FiLMGrayDeathmatch,
    "ppo_film_factorized_gray": PPOFrameStack_FiLMFactorized,
    "ppo_film_factorized_gray_cig": PPOFrameStack_FiLMFactorizedCig,
}

# Maps model to FRAME_STACK_SIZE
FRAME_STACK_SIZE = {
    "q_rainbow_stacked_rgb_center": DQN_FRAME_STACK_SIZE_RAINBOW_RGB_CENTER,
    "q_rainbow_stacked_gray_center": DQN_FRAME_STACK_SIZE_RAINBOW_GRAY_CENTER,
    "q_rainbow_stacked_gray_corridor": DQN_FRAME_STACK_SIZE_RAINBOW_GRAY_CORRIDOR,
    "ppo_cnn_gray": PPO_FRAME_STACK_SIZE_CNN_GRAY,
    "ppo_cnn_stacked_gray_corridor": PPO_FRAME_STACK_SIZE_CNN_GRAY_CORRIDOR,
    "ppo_late_fusion_rgb": PPO_FRAME_STACK_SIZE_LATE_FUSION_RGB,
    "ppo_late_fusion_rgb_corridor": PPO_FRAME_STACK_SIZE_LATE_FUSION_RGB_CORRIDOR,
    "ppo_cnn_stacked_gray_deathmatch": PPO_FRAME_STACK_SIZE_CNN_GRAY_DEATHMATCH,
    "ppo_late_fusion_gray": PPO_FRAME_STACK_SIZE_LATE_FUSION_GRAY,
    "ppo_late_fusion_gray_corridor": PPO_FRAME_STACK_SIZE_LATE_FUSION_GRAY_CORRIDOR,
    "ppo_late_fusion_gray_deathmatch": PPO_FRAME_STACK_SIZE_LATE_FUSION_GRAY_DEATHMATCH,
    "ppo_film_gray_center": PPO_FRAME_STACK_SIZE_FILM_GRAY_CENTER,
    "ppo_film_gray_corridor": PPO_FRAME_STACK_SIZE_FILM_GRAY_CORRIDOR,
    "ppo_film_gray_deathmatch": PPO_FRAME_STACK_SIZE_FILM_GRAY_DEATHMATCH,
    "ppo_film_factorized_gray": PPO_FRAME_STACK_SIZE_FILM_FACTORIZED,
    "ppo_film_factorized_gray_cig": PPO_FRAME_STACK_SIZE_FILM_FACTORIZED_CIG,
}

# Late-fusion PPO models (need state_vars in get_action)
PPO_STATE_VAR_MODELS = {
    "ppo_late_fusion_rgb",
    "ppo_late_fusion_gray_line",
    "ppo_late_fusion_rgb_line",
    "ppo_late_fusion_rgb_center",
    "ppo_late_fusion_gray_corridor",
    "ppo_late_fusion_rgb_corridor",
    "ppo_late_fusion_gray_deathmatch",
    "ppo_late_fusion_gray",
    "ppo_film_gray_center",
    "ppo_film_gray_corridor",
    "ppo_film_gray_deathmatch",
    "ppo_film_factorized_gray",
    "ppo_film_factorized_gray_cig",
}

# Lazy-stacked Rainbow models need special preprocessing
LAZY_STACK_MODULE_BY_MODEL = {
    "q_rainbow_stacked_rgb_center": rainbow_lazy_mod_rgb_center,
    "q_rainbow_stacked_gray_center": rainbow_lazy_mod_gray_center,
    "q_rainbow_stacked_gray_corridor": rainbow_lazy_mod_gray_corridor,
}

# Factorized Mapper
FACTORIZED_ACTION_MAPPER = {
    "ppo_film_factorized_gray": FactorizedActionMapper,
    "ppo_film_factorized_gray_cig": FactorizedActionMapperCig,
}