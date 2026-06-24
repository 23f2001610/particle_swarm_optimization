import numpy as np
import matplotlib.pyplot as plt

# =========================
# PARAMETERS
# =========================

SNo = 32
SSpacing = 1.8083e-3
CurRadius = 48e-3
Center = np.array([0.0, 0.0])
c = 1500.0

# =========================
# SENSOR POSITIONS
# =========================

def get_sensor_positions_imasonic(
        nSensors,
        SensorSpacing,
        CurvatureRadius,
        pFocalCenter,
        RotationAngle=0):

    phiSensors = (
        (np.arange(nSensors) - (nSensors-1)/2)
        * 2*np.arctan(SensorSpacing/(2*CurvatureRadius))
    )

    pSensors = CurvatureRadius * np.vstack((
        -np.cos(phiSensors),
         np.sin(phiSensors)
    ))

    pSensors += pFocalCenter.reshape(2,1)

    if RotationAngle != 0:

        angle = np.deg2rad(RotationAngle)

        R = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])

        pSensors = R @ pSensors

    return pSensors


DSSs = get_sensor_positions_imasonic(
    SNo,
    SSpacing,
    CurRadius,
    Center,
    0
)

# =========================
# LOAD DATA
# =========================

data = np.loadtxt("Data32DeL.dat", delimiter=",")

dt = 1.9593e-8

# =========================
# EXTRACT ARRIVAL TIMES
# =========================

arrival_sample = np.zeros(SNo)

for i in range(SNo):
    arrival_sample[i] = np.argmax(np.abs(data[i,:]))

t_measured = arrival_sample * dt
t_measured -= np.min(t_measured)

# =========================
# FITNESS FUNCTION
# =========================

def fitness_function(pos, DSSs, t_measured, c):

    d_pred = np.sqrt(
        (pos[0] - DSSs[0,:])**2 +
        (pos[1] - DSSs[1,:])**2
    )

    t_pred = d_pred / c
    t_pred -= np.min(t_pred)

    err = np.sum((t_pred - t_measured)**2)

    return err

# =========================
# PSO PARAMETERS
# =========================

numParticles = 32
maxIter = 100

w = 0.7
c1 = 1.5
c2 = 1.5

nvars = 2

lb = np.array([-0.05, -0.05])
ub = np.array([ 0.05,  0.05])

# =========================
# INITIALIZE SWARM
# =========================

pos = np.zeros((numParticles,2))
vel = np.zeros((numParticles,2))

fit = np.zeros(numParticles)

pBestPos = np.zeros((numParticles,2))
pBestFit = np.zeros(numParticles)

for i in range(numParticles):

    # MATLAB:
    # pos(i,:) = DSSs(:,i)'

    pos[i,:] = DSSs[:,i]

    fit[i] = fitness_function(
        pos[i,:],
        DSSs,
        t_measured,
        c
    )

    pBestPos[i,:] = pos[i,:]
    pBestFit[i] = fit[i]

# =========================
# GLOBAL BEST
# =========================

idx = np.argmin(pBestFit)

globalBestFit = pBestFit[idx]
gBestPos = pBestPos[idx].copy()

# =========================
# LIVE PLOT
# =========================

plt.ion()

fig, ax = plt.subplots()

ax.plot(
    DSSs[0,:]*1000,
    DSSs[1,:]*1000,
    'bo'
)

particlePlot = ax.scatter(
    pos[:,0]*1000,
    pos[:,1]*1000,
    s=50
)

gBestPlot, = ax.plot(
    gBestPos[0]*1000,
    gBestPos[1]*1000,
    'kp',
    markersize=15
)

ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')

ax.set_title('PSO Particle Movement')

ax.grid(True)
ax.set_aspect('equal')

ax.set_xlim([-50,50])
ax.set_ylim([-50,50])

# =========================
# MAIN PSO LOOP
# =========================

for iteration in range(maxIter):

    for i in range(numParticles):

        r1 = np.random.rand(nvars)
        r2 = np.random.rand(nvars)

        vel[i,:] = (
            w*vel[i,:]
            + c1*r1*(pBestPos[i,:]-pos[i,:])
            + c2*r2*(gBestPos-pos[i,:])
        )

        pos[i,:] += vel[i,:]

        pos[i,:] = np.maximum(pos[i,:], lb)
        pos[i,:] = np.minimum(pos[i,:], ub)

        fit[i] = fitness_function(
            pos[i,:],
            DSSs,
            t_measured,
            c
        )

        if fit[i] < pBestFit[i]:

            pBestFit[i] = fit[i]
            pBestPos[i,:] = pos[i,:]

    idx = np.argmin(pBestFit)

    currentBestFit = pBestFit[idx]

    if currentBestFit < globalBestFit:

        globalBestFit = currentBestFit
        gBestPos = pBestPos[idx].copy()

    particlePlot.set_offsets(pos*1000)

    gBestPlot.set_xdata([gBestPos[0]*1000])
    gBestPlot.set_ydata([gBestPos[1]*1000])

    ax.set_title(
        f"PSO Iteration {iteration+1}   "
        f"Best Fitness = {globalBestFit:.3e}"
    )

    plt.draw()
    plt.pause(0.09)

    print(
        f"Iteration {iteration+1} "
        f"Best Fitness = {globalBestFit:.10e}"
    )

# =========================
# RESULT
# =========================

bestPos = gBestPos

print("\nEstimated Bird Position (mm)")
print(bestPos * 1000)

# =========================
# FINAL PLOT
# =========================

plt.ioff()

plt.figure()

plt.plot(
    DSSs[0,:]*1000,
    DSSs[1,:]*1000,
    'ro'
)

plt.plot(
    bestPos[0]*1000,
    bestPos[1]*1000,
    'b*',
    markersize=15
)

plt.xlabel("x-position [mm]")
plt.ylabel("y-position [mm]")

plt.legend([
    "Sensors",
    "Estimated Bird"
])

plt.grid(True)
plt.axis('equal')

plt.show()