# CANOPy Occlusion Mapping

This module contains the main functions to perform occlusion mapping using solely a point cloud. It is an improved version of [Gassilloud et al. (2025)](https://www.sciencedirect.com/science/article/pii/S1569843225001402), changes are noted [here](#changelog).


## Content
1. [Occlusion mapping modules](#occlusion-mapping-modules)
2. [Tutorial](#tutorial)
3. [Changelog](#changelog)


## Occlusion mapping modules
1. [Sensor Position Reconstruction](./sensor_position_reconstruction.py): Reconstructing LiDAR sensor position data from a point cloud.
2. [Pulse Origin Reconstruction](./pulse_origin_reconstruction.py): Reconstructing the pulse origin of laser pulses and beam trajectories using the reconstructed sensor position.
3. [Occlusion Mapping](./occlusion_mapping.py): Perform occlusion mapping with the reconstructed rays.


## Tutorial
A [tutorial.ipynb](./tutorial.ipynb) is provided as a detailed guide, including a link to example data and exemplary visualizations.


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
   * optional extrapolation of sensor positions to cover `gps_time` range of point cloud in [sensor position reconstruction](./sensor_position_reconstruction.py)