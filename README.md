[![PyPI version](https://badge.fury.io/py/vizdoom.svg)](https://badge.fury.io/py/vizdoom) [![Build and test](https://github.com/Farama-Foundation/ViZDoom/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/Farama-Foundation/ViZDoom/actions/workflows/build-and-test.yml) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
  <img src="https://raw.githubusercontent.com/Farama-Foundation/ViZDoom/master/docs/_static/img/vizdoom-text.png" width="500px"/>
</p>

ViZDoom allows developing AI **bots that play Doom using only visual information** (the screen buffer). It is primarily intended for research in machine visual learning, and deep reinforcement learning, in particular.

ViZDoom is based on [ZDoom](https://zdoom.org) engine to provide the game mechanics.

![ViZDoom Demo](https://raw.githubusercontent.com/Farama-Foundation/ViZDoom/master/docs/_static/img/vizdoom-demo.gif)

# Acknowledgement

This is a student project for CSC 480 at Cal Poly San Luis Obispo under Dr. Rodrigo Canaan (rcanaan@calpoly.edu).

**Members**:

- Sean Nguyen
- Eric Lee
- Jason Skeoch
- Ryan Vo
- Andrew Loader
- Deja Dominguez

To replicate this project locally, follow the steps below.

## Setting it up

Simply clone this repo and install required dependencies. Follow the ViZDoom installation guide in python [here](https://vizdoom.farama.org/introduction/python_quickstart/).

**Note**: Our AI Agents were created using a specific version of ViZDoom. For compatibility, you must install ViZDoom version 1.3.0.dev4. Otherwise, you will not be able to run the pre-trained agents.

To test that the environment has successfully been created, run the following commands:

```
# From the repo root directory
cd examples/python

# Run basic script
python3 basic.py
```

If the environment setup is correct, you should see a Doom window pop up and various logs in the console.

## Ruuning our Agents

All commands must be executed from the `/training` directory.

The `eval_models.py` script can be used to run our trained models. Running this with the `--show` (`-s`) flag set to `false` runs it asynchronously. You will not be able to see the game, but it runs much more quickly, allowing you to evaluate the performance over many episodes. Running with the flag set to `true` let's you visualize the agent in the game.

The script can be run from the CLI:

```
python3 eval_models.py -sc <scenario_name> -n <number_of_episodes> -s <show true/false> -mt <model_type> -mp ../models/<scenario_name>/<weights_filename>.pth
```

- `-sc` is the scenario name. There are 4 options: `defend_the_line`, `defend_the_center`, `deadly_corridor`, and `deathmatch`
- `-n` specifies how many episodes or games to run. Usually, 10 is sufficient
- `-s` specifies whether to show the game or not. To see the agent play, use `-s true`
- `-mt` specifies the model type. See `/training/model_registry.py` and `/training/mass_eval_manifest.json` for a list of options.
- `-mp` specifies the weights filepath. They follow this structure `/models/<scenario_name>/<weights_filename>.pth`, so when you run this script, you must use a filepath that goes to the `/model/` directory. Hence, we recommend you use `../models/..`

### Defend the Line

Script execution template:

```
python3 eval_models.py -sc defend_the_line -n 10 -s true -mt <model_type> -mp ../models/defend_the_line/<weights_filename>.pth
```

To see which model types and weight filenames you can use, please see `/training/mass_eval_manifest.json`. Note: models with a `"skip": "True"` entry are currently broken/incompatiable with the execution script, so they cannot be run.

For example, to run the `q_cnn` model type, run this command:

```
python3 eval_models.py -sc defend_the_line -n 10 -s true -mt q_cnn -mp ../models/defend_the_line/q_cnn.pth
```

### Defend the Center

Script execution template:

```
python3 eval_models.py -sc defend_the_center -n 10 -s true -mt <model_type> -mp ../models/defend_the_center/<weights_filename>.pth
```

For example, to run the `ppo_late_fusion_gray_center` model type, run this command:

```
python3 eval_models.py -sc defend_the_center -n 10 -s true -mt ppo_late_fusion_gray -mp ../models/defend_the_center/ppo_late_fusion_gray.pth
```

### Deadly Corridor

Script execution template:

```
python3 eval_models.py -sc deadly_corridor -n 10 -s true -mt <model_type> -mp ../models/deadly_corridor/<weights_filename>.pth
```

For example, to run the `ppo_film_gray_corridor` model type, run this command:

```
python3 eval_models.py -sc deadly_corridor -n 10 -s true -mt ppo_film_gray_corridor -mp ../models/deadly_corridor/ppo_film_gray.pth
```

### Deathmatch

Script execution template:

```
python3 eval_models.py -sc deathmatch -n 10 -s true -mt <model_type> -mp ../models/deathmatch/<weights_filename>.pth
```

For example, to run the `ppo_late_fusion_gray_center` model type, run this command:

```
python3 eval_models.py -sc deathmatch -n 10 -s true -mt ppo_film_gray_deathmatch -mp ../models/deathmatch/ppo_film_gray.pth
```

_Note: for this model type, you need to import the weight file from the Google Drive._

### Baseline

You can run a baseline model in each scenario. This model makes random actions across every possible action combination, serving as a comparison to our trained models.

Run this command to get the baseline model, where `<scenario>` is one of the four scenarios:

```
python3 eval_models.py -sc deathmatch -n 10 -s true -mt random
```

## Additional Resources

We have included demo runs for each scenario, including one video for the best agent and one video for the baseline model (making random actions). These can be found in this [Google Drive Folder](https://drive.google.com/drive/folders/1M7n5V2MVEMQJHNlJ4oXoK6Q-FOsC_zDB?usp=sharing)

This folder also includes the weight files for Deathmatch PPO FiLM in grayscale (`ppo_film_gray.pth`) and PPO LateFusion in grayscale (`ppo_late_fusion_gray.pth`). These weights were too large to be uploaded to the GitHub repo, so they must be accessed here instead. Simply download the both files and move them to `/models/deathmatch/`.

Finally, the results of our model evaluation over 1000-3000 episodes can be found in the `model_eval_data_<scenario>.xlsx` spreadsheets. Note, the file names use an abbreviation of the scenarios, as specified below:

- dtl = defend_the_line
- dtc = defend_the_center
- dc = deadly_corridor
- dm = deathmatch

## ViZDoom Credit

Below is the remainder of the README from the ViZDoom project, including an overview of the ViZDoom project, installation guides, and additional information. This project was built on top of the Farma Foundation's ViZDoom, which included the groundwork environment, various Doom scenarios, and examples of how to build and train AI agents.

# ViZDoom Original README

## Features

- API for Python (including [Gymnasium](https://gymnasium.farama.org/)/Gym wrappers) and C++,
- Multi-platform (Linux, macOS, Windows),
- Fast (up to 7000 frames/steps per second in sync mode, single-threaded on a modern CPU),
- Lightweight (few MBs),
- Easy-to-create custom scenarios (visual editors, powerful scripting language, and examples available),
- Async and sync single-player and multiplayer modes,
- Customizable resolution and rendering parameters,
- Access to the depth buffer (3D vision),
- Automatic labeling and categorization of game objects visible in the frame,
- Access to the audio buffer,
- Access to the list of actors/objects and map geometry,
- Access to in-game text messages and notifications,
- Off-screen rendering,
- Episodes recording,
- In-game time scaling in async mode.

ViZDoom API is **reinforcement learning** friendly (suitable also for learning from demonstration, apprenticeship learning or apprenticeship via inverse reinforcement learning, etc.).

## Cite as

> M Wydmuch, M Kempka & W Jaśkowski, ViZDoom Competitions: Playing Doom from Pixels, IEEE Transactions on Games, vol. 11, no. 3, pp. 248-259, 2019
> ([arXiv:1809.03470](https://arxiv.org/abs/1809.03470))

```
@article{Wydmuch2019ViZdoom,
  author  = {Marek Wydmuch and Micha{\l} Kempka and Wojciech Ja\'skowski},
  title   = {{ViZDoom} {C}ompetitions: {P}laying {D}oom from {P}ixels},
  journal = {IEEE Transactions on Games},
  year    = {2019},
  volume  = {11},
  number  = {3},
  pages   = {248--259},
  doi     = {10.1109/TG.2018.2877047},
  note    = {The 2022 IEEE Transactions on Games Outstanding Paper Award}
}
```

or/and

> M. Kempka, M. Wydmuch, G. Runc, J. Toczek & W. Jaśkowski, ViZDoom: A Doom-based AI Research Platform for Visual Reinforcement Learning, IEEE Conference on Computational Intelligence and Games, pp. 341-348, Santorini, Greece, 2016 ([arXiv:1605.02097](http://arxiv.org/abs/1605.02097))

```
@inproceedings{Kempka2016ViZDoom,
  author    = {Micha{\l} Kempka and Marek Wydmuch and Grzegorz Runc and Jakub Toczek and Wojciech Ja\'skowski},
  title     = {{ViZDoom}: A {D}oom-based {AI} Research Platform for Visual Reinforcement Learning},
  booktitle = {IEEE Conference on Computational Intelligence and Games},
  year      = {2016},
  address   = {Santorini, Greece},
  month     = {Sep},
  pages     = {341--348},
  publisher = {IEEE},
  doi       = {10.1109/CIG.2016.7860433},
  note      = {The Best Paper Award}
}
```

## Python quick start

### Linux

To install the latest release of ViZDoom, just run:

```sh
pip install vizdoom
```

Both x86-64 and AArch64 (ARM64) architectures are supported.
Wheels are available for Python 3.9+ on Linux.

⚠️ To use audio features, you need OpenAL install in your system.
On apt-based distros (Ubuntu, Debian, Linux Mint, etc.)

```sh
apt install libopenal-dev
```

On dnf/yum-based distros (Fedora, RHEL, CentOS, Alma/Rocky Linux, etc.)

```sh
dnf install openal-soft-devel
```

If Python wheel is not available for your platform (Python version <3.9, distros below manylinux_2_28 standard), pip will try to install (build) ViZDoom from the source.
ViZDoom requires a C++11 compiler, CMake 3.12+, Boost 1.54+ SDL2, OpenAL (optional) to install from source.
See [documentation](https://vizdoom.farama.org/introduction/python_quickstart/) for more details.

### macOS

To install the latest release of ViZDoom, just run:

```sh
pip install vizdoom
```

Since 1.3.0+, pre-build wheels are available only for Apple Silicon (M-series chips) macOS 14.0+.

⚠️ To install pre-build wheels on Intel macOS 13.0+, you need to install version 1.2.4 using pip:

```sh
pip install vizdoom==1.2.4
```

If Python wheel is not available for your platform (Python version <3.9, older macOS version), pip will try to install (build) ViZDoom from the source.
ViZDoom requires a C++11 compiler, CMake 3.12+, Boost 1.54+ SDL2, OpenAL (optional) to install from source.
See [documentation](https://vizdoom.farama.org/introduction/building/) for more details how to install dependencies.

### Windows

To install the latest release of ViZDoom, just run:

```sh
pip install vizdoom
```

At the moment, only x86-64 architecture is supported on Windows.
Wheels are available for Python 3.9+ on Windows.

Please note that the Windows version is not as well-tested as Linux and macOS versions.
It can be used for development and testing but if you want to conduct serious (time and resource-extensive) experiments on Windows,
please consider using [Docker](https://docs.docker.com/docker-for-windows/install/) or [WSL](https://docs.microsoft.com/en-us/windows/wsl) with Linux version.

### Gymnasium/Gym wrappers

Gymnasium environments are installed along with ViZDoom and are available on all platforms.
See [documentation](https://github.com/Farama-Foundation/ViZDoom/blob/master/doc/Gymnasium.md) and [examples](https://github.com/Farama-Foundation/ViZDoom/blob/master/examples/python/gymnasium_wrapper.py) on the use of Gymnasium API.

## Examples

- [Python](https://github.com/Farama-Foundation/ViZDoom/blob/master/examples/python) (contain learning examples implemented in PyTorch, TensorFlow, and Theano)
- [C++](https://github.com/Farama-Foundation/ViZDoom/blob/master/examples/c%2B%2B)

Python examples are currently the richest, so we recommend looking at them, even if you plan to use C++.
The API is almost identical between the languages, with the only difference being that Python uses snake_case and C++ camelCase for methods and functions.

## Original Doom graphics

Unfortunately, we cannot distribute ViZDoom with original Doom graphics.
If you own original Doom and Doom 2 games, you can replace [Freedoom](https://freedoom.github.io/) graphics by placing `doom2.wad` into your working directory or `vizdoom` package directory.

Alternatively, any base game WAD (including other Doom engine-based games and custom/community games) can be used by pointing to it with the [`set_doom_game_path/setDoomGamePath`](https://vizdoom.farama.org/main/api/python/doom_game/index.html#vizdoom.DoomGame.set_doom_game_path) method.

## Documentation

Detailed descriptions of all ViZDoom types and methods can be found in the [documentation](https://vizdoom.farama.org/).

Full documentation of the ZDoom engine and ACS scripting language can be found on
[ZDoom Wiki](https://zdoom.org/wiki/).

Useful articles (for advanced users who want to create custom environments/scenarios):

- [ZDoom Wiki: ACS (scripting language)](https://zdoom.org/wiki/ACS)
- [ZDoom Wiki: CVARs (console variables)](https://zdoom.org/wiki/CVARs)
- [ZDoom Wiki: CCMD (console commands)](https://zdoom.org/wiki/CCMDs)

## Awesome Doom tools/projects

- [SLADE3](http://slade.mancubus.net/) - Great Doom map (scenario) editor for Linux, MacOS and Windows.
- [Doom Builder 2](http://www.doombuilder.com/) - Another great Doom map editor for Windows.
- [OBLIGE](http://oblige.sourceforge.net/) - Doom random map generator and [PyOblige](https://github.com/mwydmuch/PyOblige) is a simple Python wrapper for it.
- [Omgifol](https://github.com/devinacker/omgifol) - Nice Python library for manipulating Doom maps.
- [NavDoom](https://github.com/agiantwhale/navdoom) - Maze navigation generator for ViZDoom (similar to DeepMind Lab).
- [MazeExplorer](https://github.com/microsoft/MazeExplorer) - A more sophisticated maze navigation generator for ViZDoom.
- [Sample Factory](https://github.com/alex-petrenko/sample-factory) - A high-performance reinforcement learning framework for ViZDoom.
- [EnvPool](https://github.com/sail-sg/envpool/) - A high-performance vectorized environment for ViZDoom.
- [Obsidian](https://github.com/dashodanger/Obsidian) - Doom random map generator, a continuation of OBLIGE.
- [LevDoom](https://github.com/TTomilin/LevDoom) - Generalization benchmark in ViZDoom featuring difficulty levels in visual complexity.
- [COOM](https://github.com/TTomilin/COOM) - Continual learning benchmark in ViZDoom offering task sequences with diverse objectives.
- [HASARD](https://github.com/TTomilin/HASARD) - A safe reinforcement learning benchmark in ViZDoom

If you have a cool project that uses ViZDoom or could be interesting to ViZDoom community, feel free to open PR to add it to this list!

## Contributions

This project is maintained and developed in our free time. All bug fixes, new examples, scenarios, and other contributions are welcome! We are also open to feature ideas and design suggestions.

We have a roadmap for future development work for ViZDoom available [here](https://github.com/Farama-Foundation/ViZDoom/issues/546).

## License

The code original to ViZDoom is under MIT license. ZDoom uses code from several sources with [varying licensing schemes](http://zdoom.org/wiki/license).
