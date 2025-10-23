"""
Construction of config file.
--------------------------
author: Matthias Gassilloud
date: 21.10.2025
--------------------------
This module runs checks on input data and creates a config file to be used
in sensor_position_reconstruction.py. This module is part of Gassilloud et al. (2025) [1].

References:

[1] Gassilloud, M., Koch, B., & Göritz, A. (2025). Occlusion mapping reveals the impact of flight and sensing parameters on vertical forest structure exploration with cost-effective UAV based laser scanning. International Journal of Applied Earth Observation and Geoinformation, 139, 104493.

"""

import sys
import os
import warnings
import yaml
from pathlib import Path
from rasterio.crs import CRS

current_dir = Path(__file__).parent.parent.parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from CANOPy.geos_utils.data_management.data_management_tb import check_dir_exists, check_file_exists, mkdir_if_missing


def create_sensor_position_reconstruction_config(config_file):


    ### read yml   
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    cfg = {}

    for k, v in config.items():  # copy
        cfg[k] = v


    ### check files
    check_dir_exists(cfg["root_dir"])
    check_file_exists(cfg["point_cloud_path"])


    ### check crs args
    try:
        crs = CRS.from_epsg(cfg['epsg_code'])
    except:
        print(f"{cfg['epsg_code']} does not seem to be valid")

    try:
        unit = crs.linear_units
        if not unit.lower() in ('metre', 'meter', 'm'):
            raise Exception
    except:
        print(f"Unit: '{unit}' of crs needs to be metric.")


    ### path management
    base_dir = os.path.join(cfg["root_dir"], "sensor_position_reconstruction")
    mkdir_if_missing(base_dir)
    cfg['position_reconstruction'] = os.path.join(base_dir, 'sensor_position_reconstruction.gpkg')


    ### check reconstruction args
    if cfg["sensor_position_reconstruction_kwargs"]["positions_per_second"] <= 0:
        raise ValueError("Positions per second cannot be <= 0.")

    if cfg["sensor_position_reconstruction_kwargs"]["positions_per_second"] > 200:
        warnings.warn("Ensure you have enough sampling frequency to reconstruct > 200 sensor positions per second.")

    if cfg["sensor_position_reconstruction_kwargs"]["traj_number_min"] <= 0:
        raise ValueError("Minimum trajectories cannot be <= 0.")

    if cfg["sensor_position_reconstruction_kwargs"]["distance_max"] <= 0:
        raise ValueError("Maximum distance between returns and reconstructed sensor position cannot be <= 0.")

    if 0 < cfg["sensor_position_reconstruction_kwargs"]["distance_max"] <= 0.001:
        warnings.warn("Very small maximum distance between trajectory and reconstructed sensor position, some positions might be filtered out.")


    return cfg