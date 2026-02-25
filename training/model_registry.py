# -----------------------------------------------------------------------------
# Agent Imports
# -----------------------------------------------------------------------------

from q_late_fusion import DQNAgent as DQNAgent_LateFusion
from q_late_fusion_rgb import DQNAgent as DQNAgent_LateFusionRGB
from q_cnn import DQNAgent as DQNAgent_CNN
from q_cnn_rgb import DQNAgent as DQNAgent_CNNRGB
from q_rainbow_rgb import DQNAgent as DQNAgent_RainbowRGB
from q_rainbow_stacked import DQNAgent as DQNAgent_RainbowLazyStack
import q_rainbow_stacked as rainbow_lazy_mod
from ppo_cnn import PPOAgent
from ppo_cnn_gray import PPOAgent as PPOAgent_Gray
from ppo_cnn_gray import FrameStack as PPOFrameStack
from ppo_cnn_gray import FRAME_STACK_SIZE as PPO_FRAME_STACK_SIZE
from ppo_cnn_rgb_center import PPOAgent as PPOAgent_CNNRGBCenter
from ppo_late_fusion_rgb_center import PPOAgent as PPOAgent_LateFusionRGBCenter
from ppo_cnn_deadly_corridor_gray import PPOAgent as PPOAgent_CNNDeadlyCorridorGray

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
}

PPO_MODELS = {
    "ppo_cnn",
    "ppo_cnn_gray",
    "ppo_cnn_rgb_center",
    "ppo_late_fusion_rgb_center",
    "ppo_cnn_deadly_corridor_gray",
}

FRAME_STACK_MODELS = {
    "ppo_cnn_gray",
    "q_rainbow_stacked",
}

# Lazy-stacked Rainbow models need special preprocessing
LAZY_STACK_MODULE_BY_MODEL = {
    "q_rainbow_stacked": rainbow_lazy_mod,
}