# CANOPy - Sensor Position Reconstruction

This module contains the main functions to perform sensor position reconstruction using solely a point cloud. It is an improved version of [Gassilloud et al. (2025)](https://www.sciencedirect.com/science/article/pii/S1569843225001402), changes are noted [here](#changelog). Licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

<br>


## Content
1. [Citation](#citation)
2. [Sensor Position Reconstruction modules](#sensor-position-reconstruction-modules)
2. [Tutorial](#tutorial)
3. [Changelog](#changelog)

<br>


## Citation
 If you find this tool usefull, please consider citing our paper:

```bibtex
@article{gassilloud2025occlusion,
title = {Occlusion mapping reveals the impact of flight and sensing parameters on vertical forest structure exploration with cost-effective UAV based laser scanning},
journal = {International Journal of Applied Earth Observation and Geoinformation},
volume = {139},
pages = {104493},
year = {2025},
issn = {1569-8432},
doi = {https://doi.org/10.1016/j.jag.2025.104493},
url = {https://www.sciencedirect.com/science/article/pii/S1569843225001402},
author = {Matthias Gassilloud and Barbara Koch and Anna Göritz}
}
```

<br>


## Sensor Position Reconstruction modules
  * [Config File](./sensor_position_reconstruction.yml): User parameters required as input.
  * [Sensor Position Reconstruction](./sensor_position_reconstruction.py): Reconstructing LiDAR sensor position data from a point cloud.
  * [Config](./config.py): Validation of user parameters and generatin of config file
  * [Example](./example/): Working example with tutorial.

<br>


## Installation
Follow the installation instructions [here](../README.md#installation).

<br>


## Tutorial
A [tutorial.ipynb](./example/tutorial.ipynb) is provided as a detailed guide, including a link to example data and exemplary visualizations.

<br>


## Changelog

### [0.1.0] - 2025-07-22

_First release with improvements implemented towards [Gassilloud et al. (2025)](https://www.sciencedirect.com/science/article/pii/S1569843225001402)_

#### Added
   * configuration with .yaml files
   * user argument validation and error handling
   * README.md documentation
   * jupyter notebook tutorial
   * function docstrings