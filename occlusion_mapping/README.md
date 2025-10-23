# CANOPy Occlusion Mapping

This module contains the main functions to perform occlusion mapping using solely a point cloud. It is an improved version of [Gassilloud et al. (2025)](https://www.sciencedirect.com/science/article/pii/S1569843225001402), changes are noted [here](#changelog). Licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).



## Content
1. [Citation](#citation)
2. [Occlusion mapping modules](#occlusion-mapping-modules)
2. [Tutorial](#tutorial)
3. [Changelog](#changelog)



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


## Occlusion mapping modules
[Occlusion Mapping Config File](./occlusion_mapping.yml): User parameters required as input.
[Occlusion Mapping Module](./occlusion_mapping.py): Perform occlusion mapping using sensor position data and a point cloud
[Config](./config.py): Validation of user parameters and generatin of config file
[Example](./example/): Working example with tutorial.

additionally for separate pulse origin reconstruction:
[Pulse Origin Reconstruction Config File](./pulse_origin_reconstruction.yml): User parameters required as input.
[Pulse Origin Reconstruction Module](./pulse_origin_reconstruction.py): Reconstructing the pulse origin and last returns to represent beam trajectories.


## Installation
Follow the installation instructions [here](../README.md#installation).



## Tutorial
A [tutorial.ipynb](./example/tutorial.ipynb) is provided as a detailed guide, including a link to example data and exemplary visualizations.



## Changelog

### [0.1.0] - 2025-07-22

_First release with improvements implemented towards [Gassilloud et al. (2025)](https://www.sciencedirect.com/science/article/pii/S1569843225001402)_

#### Changed
   * number of sensor positions defines the time interval in which sensor positions are calculated. changed from fixed calculation of interval to user defined arguments in [sensor_position_reconstruction.py](./sensor_position_reconstruction.py)
   * sensor positions are calculated as the median of closest points instead of the mean of closest points, for better robustness towards outliers in [sensor_position_reconstruction.py](./sensor_position_reconstruction.py)
   * implemented scaling of coordinates and voxel grid according to used voxel `cell_size` to reduce floating point precision errors in [occlusion_mapping.py](./occlusion_mapping.py)

#### Added
   * configuration with .yaml files
   * user argument validation and error handling
   * README.md documentation
   * jupyter notebook tutorial
   * function docstrings
   * optional extrapolation of sensor positions to cover `gps_time` range of point cloud in [sensor_position_reconstruction.py](./sensor_position_reconstruction.py)