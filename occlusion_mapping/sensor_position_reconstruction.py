"""
LiDAR sensor position reconstruction.
--------------------------
author: Matthias Gassilloud
date: 03.06.2025
--------------------------
This module implements algorithms for reconstructing UAV LiDAR sensor positions 
from a point cloud by reconstructing trajectories from multiple-return laser pulses.

The algorithms are an improved version of Gassilloud et al. (2025) [1]. The docstrings were
partially completed by the Claude 3.7 Sonnet large language model (Anthropic, 2024).

References:

[1] Gassilloud, M., Koch, B., & Göritz, A. (2025). Occlusion mapping reveals the impact of flight and sensing parameters on vertical forest structure exploration with cost-effective UAV based laser scanning. International Journal of Applied Earth Observation and Geoinformation, 139, 104493.

"""

### import modules
import sys
import numpy as np
from numba import jit, prange
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd


### include modules from parent directories
from pathlib import Path
current_dir = Path(__file__).parent.parent.parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from CANOPy.geos_utils.numba_tb.numba_tb import nb_mean_axis_0, nb_median_axis_0, nb_unique_axis0, nb_float_to_string
from CANOPy.geos_utils.algorithms_tb.closest_points.closest_points import closest_points_between_lines, closest_point_between_point_and_segment
from CANOPy.geos_utils.geodata_tb.point_cloud_tb import read_las
from CANOPy.occlusion_mapping.configs.config import create_sensor_position_reconstruction_config


@jit(nopython=True, parallel={"setitem":False}, error_model='numpy')
def closest_points_from_trajectories(traj_start_point, traj_end_point, closest_points_function_njit):
    """Compute closest points and the distance between two consecutive trajectories.
    
    This function calculates the closest points between consecutive pairs of line segments
    (trajectories) defined by their start and end points. For each pair of consecutive
    trajectories, it determines the points on both trajectories that are closest to
    each other and computes the distance between them.
    
    Parameters
    ----------
    traj_start_point : numpy.ndarray
        Start points of trajectories, with shape (n, 3) where n is the number of start points
        and each point has [x, y, z] coordinates.
    traj_end_point : numpy.ndarray
        End points of trajectories, with shape (n, 3) where n is the number of end points
        and each point has [x, y, z] coordinates.
    closest_points_function_njit : callable
        Numba-compiled function that computes the closest points and distance between two segments.
        This function should take 5 parameters: 
        (point1_start, point1_end, point2_start, point2_end, clampAll)
        and return a tuple (closest_point_on_segment1, closest_point_on_segment2, distance).

    Returns
    -------
    numpy.ndarray
        Array of closest points information with shape (n-1, 7) where n is the number of points 
        in the trajectories. Each row contains:
        - The closest point on the first trajectory [x, y, z]
        - The closest point on the consecutive trajectory [x, y, z]
        - The distance between these two points

    Raises
    ------
    ValueError
        If the input trajectories have different numbers of points, contain fewer than 2 points,
        or the points don't have 3 dimensions.
        
    Notes
    -----
    This function is optimized with Numba's just-in-time compilation and parallel processing
    to efficiently handle large numbers of trajectories.
    """


    ### Checks
    if traj_start_point.shape[0] != traj_end_point.shape[0]:
        raise ValueError("Input trajectories must have the same number of points")
    
    if traj_start_point.shape[0] < 2:
        raise ValueError("Provide at least two trajectories")
        
    if traj_start_point.shape[1] != 3 or traj_end_point.shape[1] != 3:
        raise ValueError("Points must have 3 dimensions (x, y, z)")
    

    ### Preallocate array for closest points
    number_closest_points = traj_start_point.shape[0] - 1  # number of closest points
    closest_points = np.full((number_closest_points, 2*3 + 1), np.nan, dtype=np.float64)  # closest point ptA (xyz), ptB (xyz), distance - empty array to store results


    ### Loop through each pair of consecutive trajectories
    for idx in prange(number_closest_points):
        ptA, ptB, dist = closest_points_function_njit(traj_start_point[idx], traj_end_point[idx], traj_start_point[idx+1], traj_end_point[idx+1], clampAll=True)
        
        # Store results
        closest_points[idx,0:3] = ptA  # closest point first trajectory
        closest_points[idx,3:6] = ptB  # closest point second trajectory
        closest_points[idx,6] = dist  # distance


    return closest_points

def closest_points_from_trajectories_example_usage():
    """Example demonstrating the usage of closest_points_from_trajectories function.
    
    This function creates synthetic test data with random trajectories to demonstrate
    how the closest_points_from_trajectories function works. It generates two sets
    of random points and uses them as start and end points for trajectories, then
    computes the closest points between these trajectories.
    
    Returns
    -------
    None
        Prints the resulting closest points information array
        
    Notes
    -----
    This is for demonstration and testing purposes only. The function generates a
    large number of points (1,000,000) to demonstrate the performance capabilities
    of the closest_points_from_trajectories function with the Numba optimization.
    """
    
    n_points = 1000000
    traj1 = np.random.rand(n_points, 3) * 10
    traj2 = np.random.rand(n_points, 3) * 10
    result = closest_points_from_trajectories(traj1, traj2, closest_points_between_lines)
    print(result)

def extend_pulse_trajectory(first_return, last_return, extension_max=300):
    """Extend laser pulse trajectories to reconstruct sensor origin position.
    
    This function extends the vector from the last return point to the first return
    point of a laser pulse, calculating new points along this trajectory that are
    a specified distance (extension_max) from the last return. Along this trajectory is
    the position of the sensor from which the laser pulses originated.
    
    Parameters
    ----------
    first_return : numpy.ndarray
        First return points of laser pulses with shape (n, 3), where each point has
        [x, y, z] coordinates.
    last_return : numpy.ndarray
        Last return points of laser pulses with shape (n, 3), where each point has
        [x, y, z] coordinates.
    extension_max : float, default=300
        Maximum extension distance (in the same units as the coordinates) for extending
        the vector. This should approximate the typical distance from the sensor to
        the measured objects.
    
    Returns
    -------
    numpy.ndarray
        Points with shape (n, 3) along the extended trajectory with extension_max
        from last_return.
        
    Raises
    ------
    ValueError
        If first and last returns have different number of points or if the points 
        don't have 3 dimensions.
        
    Notes
    -----
    The function assumes a linear trajectory for each laser pulse. The extension
    distance should be set according to the furthest distance between the sensor
    and measured objects in the LiDAR dataset.
    """

    ### checks
    if first_return.shape[0] != last_return.shape[0]:
        raise ValueError("First and last returns must have the same number of points")
    
    if first_return.shape[1] != 3 or last_return.shape[1] != 3:
        raise ValueError("Points must have 3 dimensions (x, y, z)")

    extension_max = np.array([extension_max], dtype=np.float64)  # convert to numpy array dtype np.float64 for safe casting

    ### extend trajectory
    vec = first_return - last_return  # vector from last to first return
    vec_lengths = np.linalg.norm(vec, axis=1).reshape(-1,1)  # vector lengths
    origin_extended = last_return + vec * np.divide(extension_max, vec_lengths, out=np.zeros_like(vec_lengths), where= vec_lengths != 0)


    return origin_extended

def extend_pulse_trajectory_example_usage(origin = np.array([10.,10.,10.])):
    """Example demonstrating the usage of extend_pulse_trajectory function.
    
    This function creates synthetic test data for demonstrating how the 
    extend_pulse_trajectory function works. It generates random points around
    a specified origin, calculates vectors from the origin to these points,
    and creates first and last return points at different distances along
    these vectors. Then it tries to reconstruct the original position using
    the extend_pulse_trajectory function.
    
    Parameters
    ----------
    origin : numpy.ndarray, optional
        The 3D coordinates of the origin point to reconstruct, by default np.array([10.,10.,10.])
        
    Returns
    -------
    None
        Prints whether the origin could be successfully reconstructed
        
    Notes
    -----
    This is for demonstration and testing purposes only. The function validates
    the reconstruction by checking if the reconstructed origin matches the input origin.
    """

    max_extension = 300  # max extension
    random_points = np.random.rand(100,3) + (origin - np.array([.5, .5, .5]))  # center random points around origin
    vec = random_points - origin  # calculate vector
    vec_length = np.linalg.norm(vec, axis=1)  # calculate distance
    last_return = origin + vec * (max_extension / vec_length.reshape(-1,1))  # extend random points to distance max_extension
    first_return = origin + vec * ((max_extension / 2) / vec_length.reshape(-1,1))  # extend random points to half of max_extension

    origin_extended = extend_pulse_trajectory(first_return, last_return, max_extension=max_extension)
    print(f"Could reconstruct {origin}: ", np.allclose(origin_extended, origin))

def point_cloud_multiple_return_first_last(point_cloud_path):
    """Extract first and last return points from LAS file with multiple returns.
    
    This function reads a LAS file, identifies pulses with multiple returns,
    and extracts the first and last return points for each of these pulses.
    
    Parameters
    ----------
    las_path : str
        Path to the LAS file to be processed.
        
    Returns
    -------
    numpy.ndarray
        First return points of multiple-return pulses with shape (n, 3),
        where each point has [x, y, z] coordinates.
    numpy.ndarray
        Last return points of multiple-return pulses with shape (n, 3),
        where each point has [x, y, z] coordinates.
    numpy.ndarray
        GPS time values corresponding to each multiple-return pulse.
    list
        Time range of the point cloud: [minimum_time, maximum_time].
        
    Notes
    -----
    The function filters out single-return pulses and only processes pulses
    that have at least two returns. Points are sorted by GPS time and return
    number before processing.
    """


    ### read point cloud
    print("Read point cloud.")  # status
    dimensions = ["gps_time", "x", "y", "z", "return_number"]  # dimensions to read
    point_cloud, point_cloud_meta = read_las(file_path=point_cloud_path, dimensions=dimensions, no_points=None)  # read las all

    
    ### get unique time stamp values
    print("Sort point cloud, count unique values")  # status
    point_cloud = point_cloud[np.lexsort((point_cloud[:,4],point_cloud[:,0]))]  # sort array by gps time (last arg), return no (first arg)
    val, c = np.unique(point_cloud[:,0], return_counts=True)  # unique time stamp values


    ### checks
    if np.any(c >= 10):
        raise ValueError("Each individual pulse requires an unique gps_time. However, unique gps_time values from loaded point cloud have >= 10 returns. This might happen due to a precision loss of gps_time, e.g. when the point cloud was previously stored from cloudcompare.")


    ### exclude single returns
    print("Extend pulses with multiple returns")  # status
    mask_counts = c >= 2  # only multiple returns


    ### get first/ last pulse
    pointer_cs = np.append(np.array([0]), np.cumsum(c))  # create index "pointer" for each pulse first return, start with 0
    first_idx = pointer_cs[0:-1]  # index of first_idx pulse
    last_idx = (pointer_cs[1:] -1)  # index of last_idx pulse
        
    pointer_cs, c = None, None  # close
    
    pulse_multiple_return_first = point_cloud[first_idx[mask_counts]][:,1:4]  # get first_idx return xyz
    pulse_multiple_return_last = point_cloud[last_idx[mask_counts]][:,1:4]  # get last_idx return xyz


    return pulse_multiple_return_first, pulse_multiple_return_last, val[mask_counts], [val[0], val[-1]]

@jit(nopython=True, fastmath=True, parallel=False, cache=True, error_model='numpy')  #setitem":False})
def reconstruct_sensor_position(pulse_multiple_return_origin_extended,
                                pulse_multiple_return_last,
                                closest_points,
                                gps_time,
                                positions_per_second,
                                min_multiple_return_pulses_per_sampling_position,
                                centroid_calculation_njit):
    """Reconstruct sensor position from closest points.
    Assign time of closest pulse_multiple_return trajectory to closest sensor position.

    Parameters
    ----------
    pulse_multiple_return_origin_extended : numpy.ndarray
        Extended origin points of trajectories with shape (n, 3), where each point has [x, y, z] coordinates.
        These represent the extended starting points of laser pulse trajectories.
    pulse_multiple_return_last : numpy.ndarray
        Last return points of trajectories with shape (n, 3), where each point has [x, y, z] coordinates.
        These represent the end points of laser pulse trajectories.
    closest_points : numpy.ndarray
        Array of closest points between trajectories with shape (n, 3), 
        where each point has [x, y, z] coordinates.
    gps_time : numpy.ndarray
        Array of GPS timestamps corresponding to each trajectory, with shape (n,).
    positions_per_second : int
        Number of sensor positions to reconstruct per second of recorded data.
        Controls the temporal resolution of the reconstruction.
    min_multiple_return_pulses_per_sampling_position : int
        Minimum number of laser pulse trajectories required to reconstruct a sensor position.
        Used to ensure statistical robustness of the reconstructed positions.
    centroid_calculation_njit : callable
        Numba-compiled function that calculates the centroid from a set of points.
        Should take an array of points and return a single point [x, y, z].

    Returns
    -------
    numpy.ndarray
        Array of reconstructed sensor positions with shape (m, 4), where m is the number of
        reconstructed positions and each row contains [gps_time, x, y, z].
    numpy.ndarray
        Array of distances between each reconstructed sensor position and its closest trajectory,
        with shape (m,), used for quality assessment.

    Raises
    ------
    ValueError
        If the number of timestamps, start points and end points do not match
    ValueError
        If no sensor position could be reconstructed.
    """

    print("Reconstruct sensor position from closest points")

    ### checks
    if not (len(gps_time) == pulse_multiple_return_origin_extended.shape[0] == pulse_multiple_return_last.shape[0]):
        raise ValueError("Number of timestamps, start points and end points do not match.")


    ### create time intervals
    intervals = np.floor((gps_time - gps_time[0]) / (1 / positions_per_second))  # gps time intervals, starting with first closest points time
    _, _, inverse, c = nb_unique_axis0(intervals.reshape(-1,1), return_inverse=True, return_counts=True)  # get unique time intervals
    idx_pts_cell_sorted = np.argsort(inverse)  # get index that would result in sorted array -> points would be sorted by unique time intervals
    
    intervals, inverse = None, None  # close
    

    ### sorted by interval, no need to mask since pointer get masked
    closest_points = closest_points[idx_pts_cell_sorted]
    pulse_multiple_return_origin_extended = pulse_multiple_return_origin_extended[idx_pts_cell_sorted]
    pulse_multiple_return_last = pulse_multiple_return_last[idx_pts_cell_sorted]
    gps_time = gps_time[idx_pts_cell_sorted]  # gps times

    idx_pts_cell_sorted = None  # close


    ### mask minimum amount of closest points per interval
    mask_counts = c >= min_multiple_return_pulses_per_sampling_position


    ### checks
    reconstruction_failed = np.size(mask_counts) - np.count_nonzero(mask_counts)
    if reconstruction_failed > 0:
        print(f"Could not reconstruct {reconstruction_failed}"
        f" / {np.size(mask_counts)} positions.\n" \
        f"    Reduce sensor positions per second ({positions_per_second})"
        f" and/or min number of beams per position ({min_multiple_return_pulses_per_sampling_position}).")
    else:
        print(f"Reconstructed {np.size(mask_counts)} positions")

    if reconstruction_failed == np.size(mask_counts):
        raise ValueError("Reconstruction failed.")
    
    assert gps_time.shape[0] == pulse_multiple_return_origin_extended.shape[0] == pulse_multiple_return_last.shape[0]


    ### create pointer index, mask index, mask gps time values
    pointer_cs = np.append(np.array([0]), np.cumsum(c))  # create pointer, starting with 0
    first_idx = pointer_cs[0:-1][mask_counts]  # index of first_idx pulse
    last_idx = pointer_cs[1:][mask_counts]  # index of last_idx pulse
    c = c[mask_counts]

    pointer_cs = None  # close
 

    ### calculate sensor position for each interval
    print("Assign gps_time of closest trajectories to reconstructed sensor positions.")
    centroids_result = np.full((first_idx.shape[0], 4), np.nan)  # create storage
    all_distances = np.full(first_idx.shape[0], np.nan)  # create storage

    for idx in prange(first_idx.shape[0]):

        # create interval pointers
        interval_start_idx = first_idx[idx]  # get first interval pointer
        interval_end_idx = last_idx[idx]  # get last interval pointer
        interval_instance_numbers = c[idx]  # get number of centroids / pulse multiple return trajectories

        # calculate sensor position
        interval_closest_points = closest_points[interval_start_idx:interval_end_idx]  # get points in interval
        centroid = centroid_calculation_njit(interval_closest_points)  # sensor position
        

        # calculate distances between trajectories of time interval and centroid
        distances = np.full(interval_instance_numbers, np.nan)  # storage array

        for n_idx in prange(interval_instance_numbers):
            pmr_idx = interval_start_idx + n_idx  # calculate pointer for pulse_multiple_return trajectory

            closest_point = closest_point_between_point_and_segment(centroid,
                                                                    pulse_multiple_return_origin_extended[pmr_idx],
                                                                    pulse_multiple_return_last[pmr_idx])
            
            # calculate distances of closest points to centroid
            distances[n_idx] = np.linalg.norm(centroid - closest_point)


        # assign time from closest pulse_multiple_return trajectory
        min_idx = np.argmin(distances)  # closest distance index
        centroid_time = gps_time[interval_start_idx + min_idx]  # get time of closest distance

        # store result
        centroids_result[idx, 0] = centroid_time
        centroids_result[idx, 1:] = centroid

        all_distances[idx] = distances[min_idx]


    ### print result statistics
    number_avg_str = nb_float_to_string(np.mean(c))
    dist_avg = np.mean(all_distances)
    dist_avg_str = nb_float_to_string(dist_avg)
    print(f"Average number of trajectories for each reconstructed sensor position: {number_avg_str}")
    print(f"Average distance between closest trajectory and sensor position: {dist_avg_str}")
    if dist_avg >= 0.01:
        print(f"Average distance >= 0.01 meters, the spatial accuracy of your point cloud seems a bit noisy. " \
            "The resulting reconstructed sensor positions might be as well. Possible causes:\n" \
            "    - re-storage of coordiantes with scaled values may lead to inaccuracies (e.g. using laspy)\n" \
            "    - floating point precision (e.g. storage using cloudcompare)\n"
            "    - storage of incorrect scale/shift values\n" \
            "    - IMU/GPS inaccuracies (less likely)\n" \
            "If you cannot provide original/ good quality point cloud data, you could improve results by reducing 'positions_per_second' (median across more samples) and increase 'distance_max' (less filtering).")
        

    return centroids_result, all_distances

def extrapolate_position_by_time(time, A, B):
    """Extrapolate sensor position to a given time using linear interpolation.

    Calculates a new sensor position by linearly extrapolating between two known 
    positions (A and B) based on the target time. The function assumes a constant 
    velocity between the known positions and extends this motion to estimate the 
    position at the specified time.

    Parameters
    ----------
    time : float or numpy.ndarray
        Target time for which to extrapolate the sensor position.
    A : numpy.ndarray
        First reference point with format [time, x, y, z].
        The time component must be different from B's time.
    B : numpy.ndarray
        Second reference point with format [time, x, y, z].
        Used with A to determine the direction and rate of movement.

    Returns
    -------
    numpy.ndarray
        Extrapolated sensor position at the target time with format [time, x, y, z].
        The returned array maintains the spatial trajectory implied by points A and B,
        but with the time component set to the input time parameter.

    Notes
    -----
    The function assumes linear motion between points A and B and that the time 
    component in both A and B arrays is at index 0, followed by spatial coordinates.
    """

    ### extrapolate
    vector = B - A  # vector from A to B
    time_diff_norm = (time - A[0]) / vector[0] if vector[0] != 0 else 0  # difference A to time, normalize
    new_position = A + time_diff_norm * vector  # calculate new position
    new_position[0] = time  # assign time
    
    
    return new_position

def reconstruct_uav_sensor_trajectory(point_cloud_path, sensor_position_path, epsg, extension_max, positions_per_second=100, traj_number_min=5, distance_max=0.005, extrapolate=True):
    """Reconstruct UAV sensor trajectory from LiDAR point cloud data.
    
    This function reconstructs the sensor positions of a UAV-mounted LiDAR sensor from
    point cloud data. It identifies multiple-return laser pulses,
    extends their trajectories, finds the closest points between trajectories, and
    calculates sensor positions based on these points.
    
    Parameters
    ----------
    point_cloud_path : str
        Path to the input LAS file containing the point cloud data. The point cloud should include
        at least following information: ["gps_time", "x", "y", "z", "return_number"].
    sensor_position_path : str
        Path where the reconstructed sensor trajectory will be stored as a GeoPackage file.
    positions_per_second : int, optional
        Number of sensor positions to reconstruct per second of recorded data,
        by default 100. Controls the temporal resolution of the reconstruction.
    epsg : int, optional
        EPSG code for the coordinate reference system of the point cloud, by default "32632".
    traj_number_min : int, optional
        Minimum number of laser pulse trajectories required to reconstruct a sensor position,
        by default 5. Used to ensure statistical robustness.
    distance_max : float, optional
        Maximum allowed distance between a reconstructed sensor position and the closest reconstructed
        trajectory, by default 0.005. Used for filtering positions with poor reconstruction.
    extrapolate : bool, optional
        Whether to extrapolate sensor positions to cover the entire time range of the point
        cloud, by default True.
    
    Returns
    -------
    None
        Results are saved to the specified sensor_position_path as a GeoPackage file.
        
    Notes
    -----
    The function performs several key steps:
    1. Extracts first and last return points from multiple-return pulses
    2. Extends pulse trajectories to estimate sensor positions
    3. Calculates closest points between consecutive trajectories
    4. Reconstructs sensor positions for time intervals
    5. Filters positions by distance threshold
    6. Optionally extrapolates to cover the entire time range
    7. Saves the reconstructed trajectory as a GeoPackage file
    
    The reconstructed sensor positions are stored with columns:
    - gps_time: The GPS timestamp of the position
    - X, Y, Z: The 3D coordinates of the sensor position
    """


    ### get multiple return trajectories with corresponding gps time
    pulse_multiple_return_first, pulse_multiple_return_last, gps_time, time_min_max = point_cloud_multiple_return_first_last(point_cloud_path)  # get first/ last pulse


    ### extend trajectory to include sensor position
    pulse_multiple_return_origin_extended = extend_pulse_trajectory(first_return=pulse_multiple_return_first, last_return=pulse_multiple_return_last,  extension_max=extension_max)  # extend pulse trajectory
    pulse_multiple_return_first = None  # close


    ### calculate closest points between trajectories
    closest_points = closest_points_from_trajectories(traj_start_point=pulse_multiple_return_origin_extended.copy(), traj_end_point=pulse_multiple_return_last.copy(), closest_points_function_njit=closest_points_between_lines)
    closest_points = np.mean(np.array([closest_points[1:,:3], closest_points[:-1,3:6]]), axis=0)  # trajectory mean closest points
    

    ### match closest points_dimension
    gps_time = gps_time[1:-1]
    pulse_multiple_return_origin_extended = pulse_multiple_return_origin_extended[1:-1]
    pulse_multiple_return_last = pulse_multiple_return_last[1:-1]


    ### reconstruct sensor position
    sensor_position, distances = reconstruct_sensor_position(
        pulse_multiple_return_origin_extended,
        pulse_multiple_return_last,
        closest_points,
        gps_time,
        positions_per_second,
        traj_number_min,
        centroid_calculation_njit=nb_median_axis_0
        )
    

    ### optional: plot distance for visualization
    _, bins, _ = plt.hist(distances, bins='auto', label="histogram bins")  # arguments are passed to np.histogram
    plt.axvline(distance_max, c="red", label = "threshold: distance_max")
    plt.title("Min. dist. between reconstructed sensor position and closest trajectory")
    plt.xlabel("Closest distance (meter)")
    plt.ylabel(f"Number instances / {len(bins)-1} bins (auto)")
    plt.legend()
    plt.show(block=False)


    ### filter by min distance
    mask_distance = distances <= distance_max
    distances = None  # close

    sensor_position = sensor_position[mask_distance,:]

    distance_failed = np.size(mask_distance) - np.count_nonzero(mask_distance)
    if distance_failed > 0:
        print(f"Could not assign gps_time to {distance_failed}"
        f" / {np.size(mask_distance)} positions.\n" \
        f"    Increase distance_max ({distance_max}).")
    else:
        print(f"Assigned gps_time to {np.size(mask_distance)} positions")
    mask_distance = None



    ### extrapolate
    if extrapolate:
        assert sensor_position.shape[0] >= 2  # assert at least 2 sensor positions reconstructed
        if time_min_max[0] < sensor_position[0,0]:
            print("Extrapolate to start position.")
            first_position = extrapolate_position_by_time(time=time_min_max[0],
                                 A=sensor_position[0],
                                 B=sensor_position[1])
            sensor_position = np.vstack((first_position, sensor_position))

        if time_min_max[1] > sensor_position[-1,0]:
            print("Extrapolate to last position.")
            last_position = extrapolate_position_by_time(time=time_min_max[1],
                                 A=sensor_position[-2],
                                 B=sensor_position[-1])
            sensor_position = np.vstack((sensor_position, last_position))


    ### store sensor position as geopackage file
    print("Store file.")  # status
    sensor_position_df = pd.DataFrame(data = sensor_position, columns = ["gps_time", "X", "Y", "Z"])  # create pd df
    sensor_position_gdf = gpd.GeoDataFrame(sensor_position_df, geometry=gpd.points_from_xy(sensor_position_df.X, sensor_position_df.Y), crs=epsg)  # create gpd df
    sensor_position_gdf.to_file(sensor_position_path, driver='GPKG')  # store
    
    del sensor_position_df  # close
    del sensor_position_gdf  # close
    print("Sensor reconstruction complete.")  # status

def main():


    ### argument parser
    parser = argparse.ArgumentParser(description='Sensor position reconstruction from point cloud.')
    parser.add_argument('--config',
                        help='Config file for sensor position reconstruction')
    args = parser.parse_args()


    ### create config
    cfg = create_sensor_position_reconstruction_config(args.config)

    point_cloud_path = cfg["point_cloud_path"]
    epsg = cfg["epsg_code"]
    sensor_position_path = cfg["position_reconstruction"]
    extension_max = cfg["sensor_position_reconstruction_kwargs"]["extension_max"]
    positions_per_second = cfg["sensor_position_reconstruction_kwargs"]["positions_per_second"]
    traj_number_min = cfg["sensor_position_reconstruction_kwargs"]["traj_number_min"]
    distance_max = cfg["sensor_position_reconstruction_kwargs"]["distance_max"]
    extrapolate = cfg["sensor_position_reconstruction_kwargs"]["extrapolate"]


    ## run trajectory recosntruction
    reconstruct_uav_sensor_trajectory(point_cloud_path=point_cloud_path,
                                      sensor_position_path=sensor_position_path,
                                      epsg=epsg,
                                      extension_max=extension_max,
                                      positions_per_second=positions_per_second,
                                      traj_number_min=traj_number_min,
                                      distance_max=distance_max,
                                      extrapolate=extrapolate)

if __name__ == "__main__":
    main()
