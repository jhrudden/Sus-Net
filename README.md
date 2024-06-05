# Sus-Net (Playing Among Us with RL)

In this project we explore the application of Deep Q-Learning to the popular mobile game Among Us. We create a simulation of the game containing two teams, imposters and crew members. While crew members seek to complete tasks throughout the map, imposters aim to eliminate the crew memebrs anb sabotage tasks without being voted out. We create a simple 2D representation of the game in which agents are able to move, do jobs, sabotage, kill, and vote. Checkout our [Presentation](./assets/SusNet_Presentation.pdf) and [Final Report](./assets/CS5180_Report.pdf) for more details!

## Setup

1. Creating python virtual environment

```bash
conda env create -f environment.yml
conda activate sus_net
```

2. Setup pre-commit hooks

```bash
pre-commit install
```

3. Have fun!


## Experimental Results

### One Imposter -vs- One Crew Member

#### Empty Evironment

![no_walls_1v1](https://github.com/jhrudden/Sus-Net/assets/90374336/bb8b3d14-5e85-4b24-b280-6348c45f38dd)

#### Evironment with Walls

![walls_1v1](https://github.com/jhrudden/Sus-Net/assets/90374336/d3795621-39e5-4abf-950e-da1c08df0b55)


### One Imposter -vs- Two Crew Members

#### Empty Evironment
<p align="center">
<img src="https://github.com/jhrudden/Sus-Net/assets/90374336/d8e87499-283c-4f20-9ed4-e0a9af7080f0" width="500" height="500" alt="Not Walled">
</p>


#### Evironment with Walls
<p align="center">
<img src="https://github.com/jhrudden/Sus-Net/assets/90374336/b095a75e-1025-4252-8d2b-2eca40f99d71" width="500" height="500" alt="Walled">
</p>
