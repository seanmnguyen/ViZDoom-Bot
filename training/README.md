# Adding New Model Training

## File Naming Convention:

Pattern: `{learning_algorithm}_{architecture}_{color}.py` where

- `learning_algorithm` is `q` (Q-Learning) or `ppo` (PPO)
- `architecture` is `cnn`, `late_fusion`, or `film`
- `color` is `gray` or `rgb`

## Standardized Hyperparameters

We want to evaluate the models designs, not just how long training ran. So, we need to use the same training parameters/hyperparameters moving forward. Use:

```
# Q-learning settings
LEARNING_RATE = 0.00025
DISCOUNT_FACTOR = 0.99
TRAIN_EPOCHS = 10
LEARNING_STEPS_PER_EPOCH = 2000
REPLAY_MEMORY_SIZE = 10000

# NN learning settings
BATCH_SIZE = 128

# Training regime
TEST_EPISODES_PER_EPOCH = 100

# Other parameters
FRAME_REPEAT = 12
RESOLUTION = (96, 128)
EPISODES_TO_WATCH = 10
```

These are loaded into `utils.py`, so you can just import the file and reference the parameters directly.

**NOTE**: If you've already been working on some of these training files, you likely used `resolution=(30, 45)`. That was the default from the original repo, but this changes the image aspect ratio and downsizes thte image too much. We're using `(96, 128)` instead, so be careful with this. If you're just copying an existing file, check if it's been set up for `(30, 45)` instead since you may need to change some things (especially if you're referencing a file using grayscale).

## Saving Model Weights

Save the model weights (.pth file) to the `../models/{SCENARIO_NAME}/` folder. The model weights file should use the same name as this training file.

For example, if you are writing `training/ppo_late_fusion_rgb.py` then the name of the model weights file should be `ppo_late_fusion_rgb.pth`.

Overall, the `model_savefile` should look like:

`model_savefile = f"../models/{SCENARIO_NAME}/<ADD_NAME_HERE>.pth"`

where:

- `{SCENARIO_NAME}` is imported from `utils.py`. You do not need to change this.
- `<ADD_NAME_HERE>` is the model weights file name. You need to add this.

## Updating demo.py

After creating a new training file and training the model, you'll need to update the `demo.py` file so we can run the model.

The model type should be the name of your training file (e.g. if your training file is `ppo_late_fusion_rgb.py`, then model type should be `ppo_late_fusion_rgb`).

First, import your Agent under "AGENTS IMPORT".

Then, go to the Model Mappings section and add the following:

- MODEL_DEFAULT_SCENARIO: map your model type to whatever scenario you trained
- AGENT_BY_MODEL: map your model type to the Agent you imported
- RESOLUTION_BY_MODEL: map your model type to your resolution (this is mostly for backwards compatiability. It should be the default (96, 128))
- COLOR_BY_MODEL: map your model type to RGB or GRAYSCALE appropriately
- PPO_MODELS: add your model type here if it uses PPO learning
