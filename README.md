# CANOPy - Canopy Attenuation and Occlusion in Python

The CANOPy module contains various point cloud data processing functions in python.


![Occlusion](./occlusion.png "Occlusion")

<br>


## Tools
1. [sensor position reconstruction](./sensor_position_reconstruction/)
    * reconstruct sensor positions using solely a point cloud
2. [occlusion mapping](./occlusion_mapping/)
    * perform occlusion mapping using solely a point cloud
3. [geospatial utils](https://github.com/MGEOS/geos_utils)
    * collection of various geospatial processing and data management tools, included as a submodule

<br>


## Download and Dependencies
The CANOPy package requires the geos_utils repository as a submodule. Download this repository with:

```bash
git clone --recursive https://github.com/MGEOS/CANOPy
```

<br>


## Installation
To run this code we recommended to use python=3.10 or higher.

```bash
conda create -n canopy python=3.10
conda activate canopy
conda install numpy numba laszip laspy lazrs-python fiona shapely rasterio pyproj pandas geopandas gdal ipykernel pyyaml
```

<br>


## Citation
If you find this useful for your research, please consider citing our paper:

```bibtex
@article{gassilloud2025occlusion,
  title={Occlusion mapping reveals the impact of flight and sensing parameters on vertical forest structure exploration with cost-effective UAV based laser scanning},
  author={Gassilloud, Matthias and Koch, Barbara and Goeritz, Anna},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={139},
  pages={104493},
  year={2025},
  publisher={Elsevier}
}
```

<br>


## License
Licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

