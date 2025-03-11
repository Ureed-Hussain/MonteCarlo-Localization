import pickle

import matplotlib.pyplot as plt
import numpy as np


def world2map(pose, gridmap, map_res):
    max_y = np.size(gridmap, 0) - 1
    new_pose = np.zeros_like(pose)
    new_pose[0] = np.round(pose[0] / map_res)
    new_pose[1] = max_y - np.round(pose[1] / map_res)
    return new_pose.astype(int)


def v2t(pose):
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    tr = np.array([[c, -s, pose[0]], [s, c, pose[1]], [0, 0, 1]])
    return tr


def t2v(tr):
    x = tr[0, 2]
    y = tr[1, 2]
    th = np.arctan2(tr[1, 0], tr[0, 0])
    v = np.array([x, y, th])
    return v


def ranges2points(ranges, angles):
    # rays within range
    max_range = 80
    idx = (ranges < max_range) & (ranges > 0)
    # 2D points
    points = np.array([
        np.multiply(ranges[idx], np.cos(angles[idx])),
        np.multiply(ranges[idx], np.sin(angles[idx]))
    ])
    # homogeneous points
    points_hom = np.append(points, np.ones((1, np.size(points, 1))), axis=0)
    return points_hom


def ranges2cells(r_ranges, r_angles, w_pose, gridmap, map_res):
    # ranges to points
    r_points = ranges2points(r_ranges, r_angles)
    w_P = v2t(w_pose)
    w_points = np.matmul(w_P, r_points)
    # world to map
    m_points = world2map(w_points, gridmap, map_res)
    m_points = m_points[0:2, :]
    return m_points


def poses2cells(w_pose, gridmap, map_res):
    # covert to map frame
    m_pose = world2map(w_pose, gridmap, map_res)
    return m_pose


def init_uniform(num_particles, img_map, map_res):
    particles = np.zeros((num_particles, 4))
    particles[:, 0] = np.random.rand(num_particles) * np.size(img_map,
                                                              1) * map_res
    particles[:, 1] = np.random.rand(num_particles) * np.size(img_map,
                                                              0) * map_res
    particles[:, 2] = np.random.rand(num_particles) * 2 * np.pi
    particles[:, 3] = 1.0
    return particles


def plot_particles(particles, img_map, map_res):
    plt.matshow(img_map, cmap="gray")
    max_y = np.size(img_map, 0) - 1
    xs = np.copy(particles[:, 0]) / map_res
    ys = max_y - np.copy(particles[:, 1]) / map_res
    plt.plot(xs, ys, '.b')
    plt.xlim(0, np.size(img_map, 1))
    plt.ylim(0, np.size(img_map, 0))
    plt.show()



import numpy as np

def sample_motion_model_odometry(x_prev, u, alpha=[0.1, 0.1, 0.1, 0.1]):
    """
    Inputs:
    - x_prev: Previous pose [x, y, theta]
    - u: Odometry readings [delta_rot1, delta_trans, delta_rot2]
    - alpha: Noise parameters [alpha1, alpha2, alpha3, alpha4]
    
    Output:
    - x_new: New pose [x_new, y_new, theta_new]
    """
    delta_rot1, delta_trans, delta_rot2 = u
    # print(alpha)
    # Compute noise variances (Eq 5.7 in Probabilistic Robotics)
    var_rot1 = alpha[0] * delta_rot1**2 + alpha[1] * delta_trans**2
    var_trans = alpha[2] * delta_trans**2 + alpha[3] * (delta_rot1**2 + delta_rot2**2)
    var_rot2 = alpha[0] * delta_rot2**2 + alpha[1] * delta_trans**2
    
    # Sample noise from zero-mean Gaussians
    noisy_rot1 = delta_rot1 + np.random.normal(0, np.sqrt(var_rot1))
    noisy_trans = delta_trans + np.random.normal(0, np.sqrt(var_trans))
    noisy_rot2 = delta_rot2 + np.random.normal(0, np.sqrt(var_rot2))
    
    # Apply motion to previous pose (Eq 5.9)
    x, y, theta = x_prev
    theta_new = theta + noisy_rot1
    x_new = x + noisy_trans * np.cos(theta_new)
    y_new = y + noisy_trans * np.sin(theta_new)
    theta_new += noisy_rot2
    theta_new = theta_new % (2 * np.pi)  # Normalize angle
    
    return np.array([x_new, y_new, theta_new])





import numpy as np

def compute_weights(particles, z, likelihood_map):
    """
    Inputs:
    - particles: Array of shape (N, 4) [x, y, theta, weight]
    - z: Lidar measurements (2, 37) [angles, ranges]
    - gridmap: Occupancy grid map (for coordinate conversion)
    - likelihood_map: Precomputed map of measurement probabilities
    - map_res: Map resolution (meters/pixel)
    
    Output:
    - weights: Normalized importance weights (N,)
    """
    angles = z[0, :]  # Lidar beam angles (phi_i)
    ranges = z[1, :]  # Lidar ranges (rho_i)
    num_particles = particles.shape[0]
    weights = np.zeros(num_particles)
    map_res=0.1
    for i in range(num_particles):
        # Get particle pose [x, y, theta]
        pose = particles[i, :3]
        # Convert lidar ranges to map cells for this particle
        m_points = ranges2cells(ranges, angles, pose, likelihood_map, map_res)
        
        # Clip coordinates to map boundaries
        m_x = np.clip(m_points[0, :].astype(int), 0, likelihood_map.shape[1]-1)
        m_y = np.clip(m_points[1, :].astype(int), 0, likelihood_map.shape[0]-1)
        
        # Compute log-likelihood to avoid underflow
        log_weight = np.sum(np.log(likelihood_map[m_y, m_x] + 1e-10))  # Add epsilon to avoid log(0)
        weights[i] = np.exp(log_weight)
    
    # Normalize weights
    weights /= np.sum(weights)
    return weights


def resample(particles, weights, img_map, map_res):
    """
    Resamples particles based on their weights using systematic resampling.
    Ensures resampled particles are within the map bounds.
    """
    N = len(particles)
    indices = np.zeros(N, dtype=int)
    
    # Systematic resampling
    cumulative_sum = np.cumsum(weights)
    step = 1.0 / N
    start = np.random.uniform(0, step)
    
    j = 0
    for i in range(N):
        u = start + i * step
        while u > cumulative_sum[j]:
            j += 1
        indices[i] = j
    
    # Get the resampled particles and ensure they stay within map bounds
    resampled_particles = particles[indices]
    
    for i in range(N):
        resampled_particles[i, :2] = np.clip(resampled_particles[i, :2], 0, [img_map.shape[1] * map_res, img_map.shape[0] * map_res])
    
    return resampled_particles


import matplotlib.pyplot as plt
import numpy as np
import gif
@gif.frame
def get_frame(particles,img_map,map_res):
    plt.matshow(img_map, cmap="gray")
    max_y=np.size(img_map,0)-1
    xs=np.copy(particles[:,0])/map_res
    ys=max_y-np.copy(particles[:,1])/map_res
    plt.plot(xs,ys,'.b')
    plt.xlim(0,np.size(img_map,1))
    plt.ylim(0,np.size(img_map,0))

def mc_localization(data, num_particles=5000, map_res=0.1):
    """
    Monte Carlo Localization (MCL) using a particle filter with GIF generation.
    :param data: The dataset containing map, likelihood map, odometry, and laser scans.
    :param num_particles: Number of particles for localization.
    :param map_res: Resolution of the map.
    :param gif_filename: The output filename for the GIF.
    """
    # Assuming ex is a module with functions init_uniform and plot_particles
    particles = init_uniform(num_particles, data['img_map'], map_res)
    
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots()
    plot_particles(particles, data['img_map'], map_res)  # Initial plot
    plt.draw()
    plt.pause(1)
    
    frames = []  # List to store frames for GIF
    map_res=0.1
   
    # Iterate over odometry and laser scan data, but limit to 200 frames
    for t in range(len(data['odom'])):
        print("Iteration number:",t)# Limit to 200 frames
        odom = data['odom'][t]
        z = data['z'][t]  # Laser scan data
        
        # Update particle locations based on motion model
        for i in range(num_particles):
            particles[i, :3] = sample_motion_model_odometry(particles[i, :3], odom, [0.1, 0.1, 0.1, 0.1])
        
        # Compute particle weights based on sensor observations
        weights = compute_weights(particles, z, data['likelihood_map'])
        
        # Resample particles based on weights
        particles = resample(particles, weights, data['img_map'], map_res)
        
        # Refresh plot
        ax.clear()
        plot_particles(particles, data['img_map'], map_res)
        ax.set_title(f"Step {t}")
        plt.draw()
        plt.pause(0.1)
        
        frames.append(get_frame(particles,data['img_map'],map_res))  # Pause to update plot
    gif.save(frames, "mygif.gif", duration=30)

    plt.ioff()  # Disable interactive mode
    plt.show()  # Show the final plot
    
    return particles


