# Documenting Model Stats

## Defend the Line Scenario

### Scenario Description
- Actions: TURN_LEFT, TURN_RIGHT, ATTACK, MOVE_LEFT, MOVE_RIGHT
- Game Variables: AMMO2, HEALTH
- Rewards: kills (+1), dying (-1)
- Other Configurations: unlimited ammo
- Notes: Spawn on opposite wall of square map, ranged and melee enemies

### Q-Learning

#### q_late_fusion_gray_best.pth

- Architecture: QLateFusion --> CNN (screen buffer) + MLP (`AMMO2` and `HEALTH`)
- Colors: GRAYSCALE
- Training parameters: resolution=(30, 45), epochs=10, learning_steps_per_epoch=2000, batch_size=64
- Earlier versions:
  - q_late_fusion_gray1.pth 
- Performance: Average Score = 16.2
- Trainer: Sean Nguyen

Demo:

```
python3 demo.py -mt q_late_fusion -mp ../models/defend_the_line/q_late_fusion_gray_best.pth -s True -sc defend_the_line.cfg
```

#### q_late_fusion_rgb.pth

- Architecture: QLateFusion --> CNN (screen buffer) + MLP (`AMMO2` and `HEALTH`)
- Colors: RGB24
- Training parameters: resolution=(30, 45), epochs=10, learning_steps_per_epoch=2000, batch_size=64
- Performance: Average Score = 21.1
- Trainer: Sean Nguyen

Demo:

```
python3 demo.py -mt q_late_fusion -mp ../models/defend_the_line/q_late_fusion_rgb.pth -s True -sc defend_the_line.cfg
```

#### q_cnn_gray.pth

- Architecture: QCNN --> CNN (screen buffer)
- Colors: GRAYSCALE
- Training parameters: 
- Performance: Average Score = TODO
- Trainer: Jason Skeoch

Demo:

```
python3 demo.py -mt q_cnn -mp ../models/defend_the_line/q_cnn_rgb.pth -s True -sc defend_the_line.cfg
```

#### q_cnn_rgb.pth

- Architecture: QCNN --> CNN (screen buffer)
- Colors: RGB24
- Training parameters: 
- Performance: Average Score = 20.6
- Trainer: Eric Lee

Demo:

```
python3 demo.py -mt q_cnn -mp ../models/defend_the_line/q_cnn_rgb.pth -s True -sc defend_the_line.cfg
```

### PPO Learning

#### ppo_cnn_gray.pth

- Architecture: PPO_CNN --> CNN (screen buffer)
- Colors: GRAYSCALE
- Training parameters: 
- Performance: 6.6
- Trainer: Ryan Vo, Deja Dominguez

Demo:

```
python3 demo.py -mt ppo_cnn -mp ../models/defend_the_line/ppo_cnn_gray.pth -s True -sc defend_the_line.cfg
```

#### ppo_cnn_rgb.pth

- Architecture: PPO_CNN --> CNN (screen buffer)
- Colors: RGB24
- Training parameters: 
- Performance: 17.0
- Trainer: Ryan Vo, Deja Dominguez

Demo:

```
python3 demo.py -mt ppo_cnn -mp ../models/defend_the_line/ppo_cnn_rgb.pth -s True -sc defend_the_line.cfg
```

## Defend the Center Scenario

### Scenario Description
- Actions: TURN_LEFT, TURN_RIGHT, ATTACK
- Game Variables: AMMO2, HEALTH
- Rewards: kills (+1), dying (-1)
- Other Configurations: 26 ammo
- Notes: Spawn in center of round map, ranged and melee enemies

### Q-Learning

#### q_late_fusion_gray_best.pth

- Architecture: QLateFusion --> CNN (screen buffer) + MLP (`AMMO2` and `HEALTH`)
- Colors: GRAYSCALE
- Training parameters: resolution=(96, 128), epochs=50, learning_steps_per_epoch=5000, batch_size=200
- Earlier versions:
  - q_late_fusion_gray.pth 
  - q_late_fusion_gray2.pth
  - q_late_fusion_gray3.pth
- Performance: Average Score = 7.1
- Trainer: Sean Nguyen

Demo: 

```
python3 demo.py -mt q_late_fusion -mp ../models/defend_the_center/q_late_fusion_gray_new.pth -s True -sc defend_the_center.cfg
```

#### q_late_fusion_rgb_best.pth

- Architecture: QLateFusion --> CNN (screen buffer) + MLP (`AMMO2` and `HEALTH`)
- Colors: RGB24
- Training parameters: resolution=(96, 128), epochs=50, learning_steps_per_epoch=5000, batch_size=200
- Earlier versions:
  - q_late_fusion_rgb1.pth 
  - q_late_fusion_rgb2.pth
  - q_late_fusion_rgb3.pth
- Performance: Average Score = 9.4
- Trainer: Sean Nguyen

Demo: 

```
python3 demo.py -mt q_late_fusion_rgb -mp ../models/defend_the_center/q_late_fusion_rgb_best.pth -s True -sc defend_the_center.cfg
```

### PPO Learning
