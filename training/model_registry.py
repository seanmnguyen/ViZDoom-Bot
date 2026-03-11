# -----------------------------------------------------------------------------
# Agent Imports
# -----------------------------------------------------------------------------

from q_late_fusion import DQNAgent as DQNAgent_LateFusion
from q_late_fusion_rgb import DQNAgent as DQNAgent_LateFusionRGB
from q_cnn import DQNAgent as DQNAgent_CNN
from q_cnn_rgb import DQNAgent as DQNAgent_CNNRGB
from q_rainbow_rgb import DQNAgent as DQNAgent_RainbowRGB
from q_rainbow_stacked import DQNAgent as DQNAgent_RainbowLazyStack
from q_rainbow_stacked import FrameStack as DQNFrameStackRainbow
from q_rainbow_stacked import FRAME_STACK_SIZE as DQN_FRAME_STACK_SIZE_RAINBOW
import q_rainbow_stacked as rainbow_lazy_mod
from ppo_cnn import PPOAgent
from ppo_cnn_gray import PPOAgent as PPOAgent_Gray
from ppo_cnn_gray import FrameStack as PPOFrameStackGray
from ppo_cnn_gray import FRAME_STACK_SIZE as PPO_FRAME_STACK_SIZE_CNN_GRAY
from ppo_cnn_rgb_center import PPOAgent as PPOAgent_CNNRGBCenter
from ppo_late_fusion_rgb_center import PPOAgent as PPOAgent_LateFusionRGBCenter
from ppo_cnn_deadly_corridor_gray import PPOAgent as PPOAgent_CNNDeadlyCorridorGray
from ppo_late_fusion_rgb import PPOAgent as PPOAgent_LateFusionRGB
from ppo_late_fusion_rgb import FrameStackRGB as PPOFrameStackRGB
from ppo_late_fusion_rgb import FRAME_STACK_SIZE as PPO_FRAME_STACK_SIZE_LATE_FUSION_RGB
from ppo_late_fusion_rgb_corridor import PPOAgent as PPOAgent_LateFusionRGBCorridor
from ppo_late_fusion_rgb_corridor import FrameStackRGB as PPOFrameStackRGBCorridor
from ppo_late_fusion_rgb_corridor import FRAME_STACK_SIZE as PPO_FRAME_STACK_SIZE_LATE_FUSION_RGB_CORRIDOR
from ppo_late_fusion_gray import PPOAgent as PPOAgent_LateFusionGray
from ppo_late_fusion_gray import FrameStack as PPOFrameStack_LateFusionGray
from ppo_late_fusion_gray import FRAME_STACK_SIZE as PPO_FRAME_STACK_SIZE_LATE_FUSION_GRAY
from ppo_film_gray import PPOAgent as PPOAgent_FiLMGray
from ppo_film_gray import FrameStack as PPOFrameStack_FiLMGray
from ppo_film_gray import FRAME_STACK_SIZE as PPO_FRAME_STACK_SIZE_FILM_GRAY
from ppo_film_factorized_gray import PPOAgent as PPOAgent_FiLMFactorized
from ppo_film_factorized_gray import FrameStack as PPOFrameStack_FiLMFactorized
from ppo_film_factorized_gray import FRAME_STACK_SIZE as PPO_FRAME_STACK_SIZE_FILM_FACTORIZED
from ppo_film_factorized_gray import FactorizedActionMapper

# -----------------------------------------------------------------------------
# Model registry
# -----------------------------------------------------------------------------

MODEL_DEFAULT_SCENARIO = {
    "q_cnn": "defend_the_line.cfg",
    "q_cnn_rgb": "defend_the_line.cfg",
    "q_late_fusion": "defend_the_center.cfg",
    "q_late_fusion_rgb": "defend_the_center.cfg",
    "ppo_cnn": "defend_the_line.cfg",
    "ppo_cnn_gray": "defend_the_center.cfg",
    "ppo_cnn_rgb_center": "defend_the_center.cfg",
    "ppo_late_fusion_rgb_center": "defend_the_center.cfg",
    "q_late_fusion_rgb_DC": "deadly_corridor.cfg",
    "q_rainbow_rgb": "defend_the_center.cfg",
    "q_rainbow_stacked": "defend_the_center.cfg",
    "ppo_cnn_deadly_corridor_gray": "deadly_corridor.cfg",
    "ppo_late_fusion_rgb": "defend_the_center.cfg",
    "ppo_late_fusion_rgb_corridor": "deadly_corridor.cfg",
    "ppo_late_fusion_gray": "defend_the_center.cfg",
    "ppo_film_gray": "defend_the_center.cfg",
    "ppo_film_factorized_gray": "deathmatch.cfg",
}

AGENT_BY_MODEL = {
    "q_cnn": DQNAgent_CNN,
    "q_cnn_rgb": DQNAgent_CNNRGB,
    "q_late_fusion": DQNAgent_LateFusion,
    "q_late_fusion_rgb": DQNAgent_LateFusionRGB,
    "ppo_cnn": PPOAgent,
    "ppo_cnn_gray": PPOAgent_Gray,
    "ppo_cnn_rgb_center": PPOAgent_CNNRGBCenter,
    "ppo_late_fusion_rgb_center": PPOAgent_LateFusionRGBCenter,
    "q_late_fusion_rgb_DC": DQNAgent_LateFusionRGB,
    "q_rainbow_rgb": DQNAgent_RainbowRGB,
    "q_rainbow_stacked": DQNAgent_RainbowLazyStack,
    "ppo_cnn_deadly_corridor_gray": PPOAgent_CNNDeadlyCorridorGray,
    "ppo_late_fusion_rgb": PPOAgent_LateFusionRGB,
    "ppo_late_fusion_rgb_corridor": PPOAgent_LateFusionRGBCorridor,
    "ppo_late_fusion_gray": PPOAgent_LateFusionGray,
    "ppo_film_gray": PPOAgent_FiLMGray,
    "ppo_film_factorized_gray": PPOAgent_FiLMFactorized,
}

RESOLUTION_BY_MODEL = {
    "q_cnn": (30, 45),
    "q_cnn_rgb": (96, 128),
    "q_late_fusion": (96, 128),
    "q_late_fusion_rgb": (96, 128),
    "ppo_cnn": (30, 45),
    "ppo_cnn_gray": (96, 128),
    "ppo_cnn_rgb_center": (30, 45),
    "ppo_late_fusion_rgb_center": (30, 45),
    "q_late_fusion_rgb_DC": (96, 128),
    "q_rainbow_rgb": (96, 128),
    "q_rainbow_stacked": (96, 128),
    "ppo_cnn_deadly_corridor_gray": (30, 45),
    "ppo_late_fusion_rgb": (96, 128),
    "ppo_late_fusion_rgb_corridor": (96, 128),
    "ppo_late_fusion_gray": (96, 128),
    "ppo_film_gray": (96, 128),
    "ppo_film_factorized_gray": (96, 128),
}

GRAYSCALE = "GRAY8"
RGB = "RGB24"
AUTO = "AUTO"  # Deterimine from model attributes
COLOR_BY_MODEL = {
    "q_cnn": GRAYSCALE,
    "q_cnn_rgb": RGB,
    "q_late_fusion": GRAYSCALE,
    "q_late_fusion_rgb": RGB,
    "ppo_cnn": GRAYSCALE,
    "ppo_cnn_gray": GRAYSCALE,
    "ppo_cnn_rgb_center": RGB,
    "ppo_late_fusion_rgb_center": RGB,
    "q_late_fusion_rgb_DC": RGB,
    "q_rainbow_rgb": RGB,
    "q_rainbow_stacked": AUTO,
    "ppo_cnn_deadly_corridor_gray": GRAYSCALE,
    "ppo_late_fusion_rgb": RGB,
    "ppo_late_fusion_rgb_corridor": RGB,
    "ppo_late_fusion_gray": GRAYSCALE,
    "ppo_film_gray": GRAYSCALE,
    "ppo_film_factorized_gray": GRAYSCALE,
}

PPO_MODELS = {
    "ppo_cnn",
    "ppo_cnn_gray",
    "ppo_cnn_rgb_center",
    "ppo_late_fusion_rgb_center",
    "ppo_cnn_deadly_corridor_gray",
    "ppo_late_fusion_rgb",
    "ppo_late_fusion_rgb_corridor",
    "ppo_late_fusion_gray",
    "ppo_film_gray",
    "ppo_film_factorized_gray",
}

# Maps models to FrameStack class
FRAME_STACK_MODELS = {
    "q_rainbow_stacked": DQNFrameStackRainbow,
    "ppo_cnn_gray": PPOFrameStackGray,
    "ppo_late_fusion_rgb": PPOFrameStackRGB,
    "ppo_late_fusion_rgb_corridor": PPOFrameStackRGBCorridor,
    "ppo_late_fusion_gray": PPOFrameStack_LateFusionGray,
    "ppo_film_gray": PPOFrameStack_FiLMGray,
    "ppo_film_gray": PPOFrameStack_FiLMFactorized,
}

# Maps model to FRAME_STACK_SIZE
FRAME_STACK_SIZE = {
    "q_rainbow_stacked": DQN_FRAME_STACK_SIZE_RAINBOW,
    "ppo_cnn_gray": PPO_FRAME_STACK_SIZE_CNN_GRAY,
    "ppo_late_fusion_rgb": PPO_FRAME_STACK_SIZE_LATE_FUSION_RGB,
    "ppo_late_fusion_rgb_corridor": PPO_FRAME_STACK_SIZE_LATE_FUSION_RGB_CORRIDOR,
    "ppo_late_fusion_gray": PPO_FRAME_STACK_SIZE_LATE_FUSION_GRAY,
    "ppo_film_gray": PPO_FRAME_STACK_SIZE_FILM_GRAY,
    "ppo_film_gray": PPO_FRAME_STACK_SIZE_FILM_FACTORIZED,
}

# Late-fusion PPO models (need state_vars in get_action)
PPO_STATE_VAR_MODELS = {
    "ppo_late_fusion_rgb",
    "ppo_late_fusion_rgb_center",
    "ppo_late_fusion_rgb_corridor",
    "ppo_late_fusion_gray",
    "ppo_film_gray",
    "ppo_film_factorized_gray",
}

# Lazy-stacked Rainbow models need special preprocessing
LAZY_STACK_MODULE_BY_MODEL = {
    "q_rainbow_stacked": rainbow_lazy_mod,
}