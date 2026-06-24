import numpy as np
np.random.seed(42)

# =========================
# LOAD DATA (.DAT FILE)
# =========================
data = np.loadtxt("Data32DeL.dat", delimiter=",")

NSENSORS = data.shape[0]

dt = 1.9593e-8   # same as MATLAB
C = 1500.0

# =========================
# EXTRACT ARRIVAL TIMES
# (same logic as MATLAB)
# =========================

arrival_idx = np.zeros(NSENSORS)

for i in range(NSENSORS):
    arrival_idx[i] = np.argmax(np.abs(data[i, :]))

t_measured = arrival_idx * dt
t_measured = t_measured - np.min(t_measured)


# =========================
# SENSOR MODEL
# =========================

def get_sensor_positions(n, S, R):

    phi = ((np.arange(n) - (n-1)/2)
           * 2*np.arctan(S/(2*R)))

    x = -R*np.cos(phi)
    y =  R*np.sin(phi)

    return np.vstack((x, y))


# =========================
# FITNESS FUNCTION
# =========================

def fitness(particle):

    R, S = particle

    if R <= 0 or S <= 0:
        return 1e30

    sensors = get_sensor_positions(NSENSORS, S, R)

    # ---- IMPORTANT: YOU MUST SET BIRD POSITION ----
    bird = np.array([1.4094e-3, 4.3134e-3])  # change if needed

    d = np.sqrt(
        (bird[0] - sensors[0])**2 +
        (bird[1] - sensors[1])**2
    )

    t_pred = d / C
    t_pred = t_pred - np.min(t_pred)

    return np.sum((t_pred - t_measured)**2)


# =========================
# PSO PARAMETERS
# =========================

n_particles = 40
iters = 150

w = 0.7
c1 = 1.5
c2 = 1.5

# bounds (meters)
R_min, R_max = 0.03, 0.07
S_min, S_max = 0.001, 0.004


# =========================
# INITIALIZATION
# =========================

pos = np.zeros((n_particles, 2))
vel = np.zeros_like(pos)

pos[:,0] = np.random.uniform(R_min, R_max, n_particles)
pos[:,1] = np.random.uniform(S_min, S_max, n_particles)

pbest = pos.copy()
pbest_val = np.array([fitness(p) for p in pos])

gbest_idx = np.argmin(pbest_val)
gbest = pbest[gbest_idx].copy()
gbest_val = pbest_val[gbest_idx]


# =========================
# PSO LOOP
# =========================

for it in range(iters):

    for i in range(n_particles):

        r1 = np.random.rand(2)
        r2 = np.random.rand(2)

        vel[i] = (
            w * vel[i]
            + c1 * r1 * (pbest[i] - pos[i])
            + c2 * r2 * (gbest - pos[i])
        )

        pos[i] += vel[i]

        pos[i,0] = np.clip(pos[i,0], R_min, R_max)
        pos[i,1] = np.clip(pos[i,1], S_min, S_max)

        f = fitness(pos[i])

        if f < pbest_val[i]:
            pbest_val[i] = f
            pbest[i] = pos[i].copy()

            if f < gbest_val:
                gbest_val = f
                gbest = pos[i].copy()

    print(f"Iter {it+1}: error = {gbest_val:.6e}")


# =========================
# RESULT
# =========================

print("\nRecovered Parameters")
print("--------------------")
print("Radius  =", gbest[0]*1000, "mm")
print("Spacing =", gbest[1]*1000, "mm")