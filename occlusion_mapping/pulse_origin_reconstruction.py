"""
LiDAR pulse origin reconstruction.
--------------------------
author: Matthias Gassilloud
date: 03.06.2025
--------------------------
This module implements algorithms for reconstructing the pulse origin of a LiDAR beam. 
Pulse origins for multiple returns are derived by extending the trajectory to the approximate
height of sensor position during pulse emission. Pulse origins for single returns are obtained
by linear interpolation between previous and following sensor positions.

The algorithms are an improved version of Gassilloud et al. (2025) [1]. The docstrings were
partially completed by the Claude 3.7 Sonnet large language model (Anthropic, 2024).

References:

[1] Gassilloud, M., Koch, B., & Göritz, A. (2025). Occlusion mapping reveals the impact of flight and sensing parameters on vertical forest structure exploration with cost-effective UAV based laser scanning. International Journal of Applied Earth Observation and Geoinformation, 139, 104493.

"""

### import modules
import sys
import time
import numpy as np
import geopandas as gpd
import argparse


### include modules from parent directories
from pathlib import Path
current_dir = Path(__file__).parent.parent.parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from CANOPy.occlusion_mapping.configs.config import create_pulse_origin_reconstruction_config
from CANOPy.geos_utils.geodata_tb.point_cloud_tb import read_las


def extend_trajectory_to_height(first_return, last_return, height):
    """Extends the trajectory of a LiDAR pulse to a specified height.
    
    This function calculates the position where a pulse would have been at a given height
    by extrapolating the trajectory defined by the first and last return points.
    
    Parameters
    ----------
    first_return : numpy.ndarray
        Array of shape (n, 3) containing the [x,y,z] coordinates of the first returns.
    last_return : numpy.ndarray
        Array of shape (n, 3) containing the [x,y,z] coordinates of the last returns.
    height : numpy.ndarray or float
        Target height(s) to extend the trajectory to. Array should be of shape (n,).
    
    Returns
    -------
    numpy.ndarray
        Array of shape (n, 3) containing the [x,y,z] coordinates of the extended trajectory points
        at the specified height.
    """
    
    vector = first_return - last_return  # vector from last to first
    height_diff_norm = ((height - first_return[:,2]) / vector[:,2]).reshape(-1,1)  # height difference, normalize 
    new_position = first_return + height_diff_norm * vector  # calculate new position

    return new_position

def theoretical_pulse_origin(pulse_time, sensor_position_time, sensor_position):
    """Calculates the theoretical pulse origin position for given pulse times using linear interpolation.
    
    This function uses the sensor position data with timestamps to interpolate the position of the
    sensor at the time each pulse was emitted. It excludes pulses with timestamps outside
    the range of available sensor positions.
    
    Parameters
    ----------
    pulse_time : numpy.ndarray
        Array of shape (n,) containing the GPS timestamps of the pulses.
    sensor_position_time : numpy.ndarray
        Array of shape (m,) containing the GPS timestamps of the sensor positions.
    sensor_position : numpy.ndarray
        Array of shape (m, 3) containing the [x,y,z] coordinates of the sensor positions.
    
    Returns
    -------
    tuple
        A tuple containing:
        - numpy.ndarray: Array of shape (k, 3) containing the calculated pulse origin positions
          [x,y,z] for pulses with timestamps within the sensor position time range.
        - numpy.ndarray: Boolean mask of shape (n,) indicating which pulses were within
          the valid time range.
    """

    ### calculate theoretical start coordinate of pulses based on provided 
    print("Calculate theoretical start point.")  # status


    ### mask pulses out of time range
    mask_trange = (pulse_time >= np.min(sensor_position_time)) & (pulse_time <= np.max(sensor_position_time))  # build mask to exclude pulses outside sbet time range
    pulse_time_in = pulse_time[mask_trange]  # exclude vals outside sbet range

    print(f"Excluded pulses: {pulse_time.shape[0]-pulse_time_in.shape[0]} / {pulse_time.shape[0]} \
        -> {((pulse_time.shape[0]-pulse_time_in.shape[0]) / pulse_time.shape[0]) * 100 }%")
    pulse_time = None  # close

    assert pulse_time_in.shape[0] > 0  # check if any pulse time within range
    
    
    ### calculate pulse origin

    # closest previous sensor position pulse_time
    sensor_posidx = np.searchsorted(sensor_position_time, pulse_time_in, side="right") - 1  # closest previous sensor position (or exact match) for each pulse_time 

    # ensure correct indexing for last element, simple approach to avoid storage of too many large arrays for masking
    if sensor_posidx[-1] + 1 == sensor_position.shape[0]:  # last position exact match, cannot be extrapolated
        sensor_position = np.vstack((sensor_position, sensor_position[-1]))  # repeat last element to match indexing
        sensor_position_time = np.append(sensor_position_time, sensor_position_time[-1])  # repeat last element to match indexing

    # vector
    sensor_position_vector = np.subtract(sensor_position[sensor_posidx + 1], sensor_position[sensor_posidx])  # vector from previous sensor position to following sensor position
    sensor_time_vector = np.subtract(sensor_position_time[sensor_posidx + 1], sensor_position_time[sensor_posidx])  # time difference between previous and following sensor position

    # time difference
    time_diff = np.subtract(pulse_time_in, sensor_position_time[sensor_posidx])  # difference pulse_time to previous sensor position time
    pulse_time_in, sensor_position_time = None, None  # close
    
    # normalize
    time_diff_norm = np.divide(time_diff, sensor_time_vector, out=np.zeros_like(time_diff), where=sensor_time_vector!=0)  # relative time passed since sbet0
    time_diff, sensor_time_vector = None, None

    # calculate start coordinate
    start_coordinate = sensor_position[sensor_posidx] + time_diff_norm.reshape(-1,1) * sensor_position_vector  # calculate theoretical sbet start point for not exact matches

    return start_coordinate, mask_trange

def reconstruct_pulse_origin(point_cloud_path, sensor_position_path, rays_storage_path=None):
    """Reconstructs the origin and endpoint coordinates of LiDAR pulses from a point cloud
    and sensor position data.
    
    This function processes a LAS file containing LiDAR point cloud data and sensor position information
    to reconstruct the trajectory of each LiDAR pulse. If no sensor position data is available, it is recommended to reconstruct
    using sensor_position_reconstruction() in the first place. It handles both single and multiple return pulses
    using different reconstruction approaches:
    - For multiple returns: Extends the trajectory from first to last return up to the sensor height
    - For single returns: Uses temporal interpolation of sensor positions
    
    Parameters
    ----------
    point_cloud_path : str
        Path to the input LAS file containing the point cloud data. The point cloud should include
        at least following information: ["gps_time", "x", "y", "z", "return_number"].
    sensor_position_path : str
        Path to the GeoPackage file containing sensor position data with timestamps.
    array_storage_path : str, optional
        Path to save the reconstructed pulse data as NumPy arrays. If provided, the function will
        save start coordinates, end coordinates, and pulse times to this file.
    
    Returns
    -------
    tuple
        A tuple containing:
        - numpy.ndarray: Array of shape (n, 3) containing the start coordinates [x,y,z] of each pulse.
        - numpy.ndarray: Array of shape (n, 3) containing the end coordinates [x,y,z] of each pulse.
        - numpy.ndarray: Array of shape (n,) containing the GPS timestamps of each pulse.
    """



    ### read point cloud

    print("Read point cloud.")  # status
    dimensions = ["gps_time", "x", "y", "z", "return_number"]  # dimensions to read
    point_cloud, point_cloud_meta = read_las(file_path=point_cloud_path, dimensions=dimensions, no_points=None)  # read las all


    ### read sensor position

    print("Read sensor positions.")  # status
    sensor_position_gdf = gpd.read_file(sensor_position_path)  # read
    sensor_position = np.array(sensor_position_gdf[["gps_time", "X", "Y", "Z"]])  # convert to array for faster indexing
    del sensor_position_gdf  # close


    ### get unique time stamp values

    print("Sort point cloud, count unique values.")  # status
    start = time.time()
    point_cloud = point_cloud[np.lexsort((point_cloud[:,4],point_cloud[:,0]))]  # sort array by gps time (last arg), return no (first arg)
    val, c = np.unique(point_cloud[:,0], return_counts=True)  # unique time stamp values


    ### reconstruct pulse multiple return using closest height from pulse origin

    print("Reconstruct pulses with multiple returns.")  # status
    mask_counts = c >= 2  # only multiple returns

    # approximate sensor position height for each pmr
    sensor_posidx = np.searchsorted(sensor_position[:,0], val[mask_counts], side="right") - 1  # closest previous sensor position (or exact match) for each pmr_time
    pmr_height = sensor_position[sensor_posidx, 3]  # get corresponding height value of sensor position
    sensor_posidx = None  # close

    # get first/ last pulse
    pointer_cs = np.append(np.array([0]), np.cumsum(c))  # create index "pointer" for each pulse first return, start with 0
    first_idx = pointer_cs[0:-1]  # index of first_idx pulse
    last_idx = (pointer_cs[1:] -1)  # index of last_idx pulse
    pointer_cs, c = None, None  # close

    # get start/ end points
    pmr_first = point_cloud[first_idx[mask_counts]][:,1:4]  # get first_idx return xyz
    pmr_last = point_cloud[last_idx[mask_counts]][:,1:4]  # get last_idx return xyz
    psr_last = point_cloud[last_idx[~mask_counts]][:,1:4]  # get idx of pulse pulse single return  - use of last_idx is correct, for single returns idx are equal but last_idx includes the last elements. 
    point_cloud, first_idx, last_idx = None, None, None  # close

    # extend pulse to approximate sensor position height
    pmr_start = extend_trajectory_to_height(first_return=pmr_first, last_return=pmr_last, height=pmr_height)
    pmr_first, pmr_height = None, None  # close


    ### reconstruct pulse single return using linear interpolation

    print("Reconstruct pulse origin for single returns.")
    psr_start, mask_trange = theoretical_pulse_origin(pulse_time=val[~mask_counts], sensor_position_time=sensor_position[:,0], sensor_position=sensor_position[:,1:4])
    psr_last = psr_last[mask_trange]  # mask trange
    assert psr_start.shape == psr_last.shape  # assert shape matches
    sensor_position, mask_trange = None, None


    ### stack results

    print("Stack results.")
    start_coordinate = np.vstack((psr_start, pmr_start))  # start_coordinate
    psr_start, pmr_start = None, None  # close

    end_coordinate = np.vstack((psr_last, pmr_last))  # end_coordinate
    psr_last, pmr_last = None, None  # close

    pulse_time = np.hstack((val[~mask_counts][mask_trange].reshape(-1), val[mask_counts]))  # pulse_time
    val = None  # close


    ### optionally store as np array

    print("Store results.")
    if rays_storage_path is not None:

        with open(rays_storage_path, 'wb') as f:
            np.save(f, start_coordinate)
            np.save(f, end_coordinate)
            np.save(f, pulse_time)


    print("Pulse origin reconstruction complete.")  # status
    return start_coordinate, end_coordinate, pulse_time

def main():

        
    ### argument parser
    parser = argparse.ArgumentParser(description='Pulse origin reconstruction from point cloud and sensor position trajectory.')
    parser.add_argument('--config',
                        help='Config file for pulse origin reconstruction')
    args = parser.parse_args()


    cfg = create_pulse_origin_reconstruction_config(args.config)

    point_cloud_path = cfg["point_cloud_path"]
    sensor_position_path = cfg["sensor_position_path"]
    rays_storage_path = cfg['rays_path'] 

    reconstruct_pulse_origin(point_cloud_path=point_cloud_path,
                             sensor_position_path=sensor_position_path,
                             rays_storage_path=rays_storage_path)

if __name__ == "__main__":
    main()