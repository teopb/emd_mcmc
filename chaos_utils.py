# Teo Price-Broncucia 2020

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import ot

def rk4(t_0, dt, total_time, x_t0, deriv, args=None):
    x_n = np.array(x_t0)
    t_n = t_0

    trajectory = np.empty((int(total_time/dt) + 1, len(x_n)))
    trajectory[0, :] = x_n

    timesteps = np.empty(int(total_time/dt) + 1)
    timesteps[0] = t_n
    i = 1
    while i < len(trajectory):
        k_1 = dt * deriv(t_n, x_n, *args)
        k_2 = dt * deriv(t_n + dt/2, x_n + (1/2) * k_1, *args)
        k_3 = dt * deriv(t_n + dt/2, x_n + (1/2) * k_2, *args)
        k_4 = dt * deriv(t_n + dt, x_n + k_3, *args)

        x_n = x_n + (1/6) * (k_1 + 2*k_2 + 2*k_3 + k_4)
        t_n = t_n + dt

        trajectory[i, :] = x_n
        timesteps[i] = t_n
        i += 1

    return trajectory, timesteps

def rk4_step(dt, t_n, x_n, deriv, args=None):
    k_1 = dt * deriv(t_n, x_n, args)
    k_2 = dt * deriv(t_n + dt/2, x_n + (1/2) * k_1, args)
    k_3 = dt * deriv(t_n + dt/2, x_n + (1/2) * k_2, args)
    k_4 = dt * deriv(t_n + dt, x_n + k_3, args)

    x_n = x_n + (1/6) * (k_1 + 2*k_2 + 2*k_3 + k_4)

    return x_n

def rk4_adapt(t_0, total_t, x_t0, tolerance, deriv, args=None, start_dt=.001):
    x_n = x_t0
    t_n = t_0
    dt = start_dt

    trajectory = [x_n]

    while t_n < total_t:

    #   full step
        full_step = rk4_step(dt, t_n, x_n, deriv, args=args)

    #   half step
        half_dt = dt/2
        half_step = rk4_step(half_dt, t_n, x_n, deriv, args=args)
        two_half_step = rk4_step(half_dt, t_n + half_dt, half_step, deriv, args=args)

        dif = np.linalg.norm(full_step - two_half_step, np.inf)

        dt = dt * (tolerance/dif) ** (1/5)

#         If new dt is smaller than half previous we need to calculate a new x_n
        if dt < half_dt:
            x_n = rk4_step(half_dt, t_n + dt/2, half_step, deriv, args=args)
            t_n = t_n + dt

        else:
            x_n = two_half_step
            t_n = t_n + half_dt*2

        trajectory = np.append(trajectory, [x_n], axis=0)

    #   TODO write to log file in log folder

    return trajectory

def lorenz_deriv(t, state, *args):

    a = args[0]
    r = args[1]
    b = args[2]

    x = state[0]
    y = state[1]
    z = state[2]

    deriv = np.array([a*(y - x),
                     r*x - y - x*z,
                     x*y - b*z])

    return deriv

def wolf(trajectory, dt, epsilon_1, epsilon_2, theiler_dist=5, theta = np.pi/9):
    start_theta = theta

    # How many neighbors to calculate. Right now the maximum of =4*theiler_dist or 10
    if 4*theiler_dist > 10:
        num_neighbors = 4*theiler_dist
    else:
        num_neighbors = 10

    neighbors = NearestNeighbors(n_neighbors=num_neighbors)
    neighbors.fit(trajectory)
    # Generate starting point. Default at 10% into trajectory
    traj_len = len(trajectory)
    start_point = traj_len//10

    end_of_traj = False

    i = start_point
    first_point = True
    L = []
    L_prime = []

    while not end_of_traj:
        distances, indices = neighbors.kneighbors([trajectory[i]])
        # find nearest point not in theiler_dist and in existing angle
        # reset helper variables
        not_found = True
        theta = start_theta

        while (theta <= np.pi * 2) and not_found:
            neighbor_index = 1
            while not_found and (neighbor_index < num_neighbors):
                j = indices[0, neighbor_index]

                # Check Theiler distance
                if np.abs(j - i) > theiler_dist:

                    # make sure j isn't at end of trajectory
                    if j < (traj_len - 2):

                        if first_point:
                            not_found = False
                            first_point = False

                        else:
                            # TODO angle check
                            a = trajectory[old_j] - trajectory[i]
                            b = trajectory[j] - trajectory[i]
                            angle = np.arccos((a @ b)/(np.linalg.norm(a)*np.linalg.norm(b)))
                            if (angle < theta) and (np.linalg.norm(b) < epsilon_1):
                                not_found = False

                neighbor_index += 1
            theta = theta * 2

        if not_found:
            # TODO
            print("nearest neighbor not found")
            print(f"start point: {start_point}")
            print(f"current i: {i}")
            print("Indices:")
            print(indices)
            print("Distances")
            print(distances)
            return

        # calculate L distance
        active_dist = np.linalg.norm(trajectory[j] - trajectory[i])
        L.append(active_dist)
        print(f"L append {active_dist}")

        # follow trajectories until distance is greater than epsilon_2
        i += 1
        j += 1

        traj_check = True
        while traj_check:
            active_dist = np.linalg.norm(trajectory[j] - trajectory[i])
            if i > (traj_len - 2):
                print("End of trajectory")
                end_of_traj = True
                traj_check = False
            elif j > (traj_len - 2):
                traj_check = False
            elif active_dist > epsilon_2:
                print(f"epsilon_2 exceeded i={i}, j={j}")
                traj_check = False
            else:
                i += 1
                j += 1

        L_prime.append(active_dist)
        print(f"L Prime append {active_dist}")
        old_j = j

    # Compute the liapunov exponents from L and L_prime
    N = i - start_point

    lyap = np.sum(np.log2(np.divide(L_prime, L)))/(N * dt)

    print(f"Number of instervals: {len(L)}")
    return lyap

def variational_lorenz_deriv(t, state, args):

    a = args[0]
    r = args[1]
    b = args[2]

    x = state[0]
    y = state[1]
    z = state[2]

    dxx = state[3]
    dxy = state[4]
    dxz = state[5]
    dyx = state[6]
    dyy = state[7]
    dyz = state[8]
    dzx = state[9]
    dzy = state[10]
    dzz = state[11]

    deriv = np.array([a*(y - x),
                     r*x - y - x*z,
                     x*y - b*z,
                     a*(-dxx + dxy),
                     (r-z)*dxx - dxy - x*dxz,
                     y*dxx + x*dxy - b*dxz,
                     a*(-dyx + dyy),
                     (r-z)*dyx - dyy - x*dyz,
                     y*dyx + x*dyy - b*dyz,
                     a*(-dzx + dzy),
                     (r-z)*dzx - dzy - x*dzz,
                     y*dzx + x*dzy - b*dzz])

    return deriv

# return scalar value
def vect_to_scalar(point, box_diam):
    scalar = 0
    for dim in range(len(point)):
        scalar += point[dim] * (box_diam**dim)

    return scalar

# return N(epsilon)
def capacity_iteration(trajectory, epsilon, diameter):
#     calculate box diameter
    box_diam = math.ceil(diameter/epsilon)
#     make set to house occupied boxes
    occupied = set()
    for point in trajectory:
#         convert to box vector
        box_point = np.floor(point/epsilon)
#         convert to scalar
        scalar = vect_to_scalar(box_point, box_diam)
#     check if already occupied, otherwise add
        if scalar not in occupied:
            occupied.add(scalar)

    return len(occupied)

def capacity_dimension(trajectory, min_ep, iterations=10):
    diameter = np.max(np.max(trajectory, axis=0) - np.min(trajectory, axis=0))
    print(f"Diameter: {diameter:.3f}")
    eps = np.logspace(np.log(min_ep), np.log(diameter), iterations)
    Ns = np.zeros(iterations)

    for i in range(iterations):
        Ns[i] = capacity_iteration(trajectory, eps[i], diameter)

    return Ns, eps

def plot_cap_dim(Ns, eps, title=None):
    ln_Ns = np.log(Ns)
    ln_inverse_eps = np.log(1/eps)
    plt.plot(ln_inverse_eps, ln_Ns)
    plt.xlabel(r"$\log\left(\frac{1}{\epsilon}\right)$")
    plt.ylabel(r"$\log(N(\epsilon))$")
    if title is not None:
        plt.title(title)
    plt.show()

# provide tau in terms of samples, 1d data
def embed(data, tau, dim):
    embedded_data = np.empty((len(data) - (dim - 1) * tau, dim))
    for i in range(len(data) - (dim - 1) * tau):
        for j in range(dim):
            embedded_data[i, j] = data[i + j * tau]

    return embedded_data

def save_trajectory(trajectory, timestamps, filename):
    ts = timestamps.reshape(-1,1)
    output_trajectory = np.append(trajectory, ts, axis=1)
    np.savetxt(filename, output_trajectory, fmt="%.16f", delimiter=' ')

def pointwise_distance_sum(trajectory1, trajectory2):
    dif = np.array(trajectory1) - np.array(trajectory2)
    abs_dif = np.absolute(dif)
    sum_abs_dif = np.sum(abs_dif)
    return sum_abs_dif