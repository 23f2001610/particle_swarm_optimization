import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PARAMETERS
# =====================================================

SNo = 32

SSpacing = 1.8083e-3      # m
CurRadius = 48.5e-3       # m

Center = np.array([0.0, 0.0])

RotationAngle = 0

# =====================================================
# KNOWN BIRD POSITION
# =====================================================

birdPos = np.array([1.4094e-3, 4.3134e-3])

# =====================================================
# SENSOR GEOMETRY FUNCTION
# =====================================================

def get_sensor_positions_imasonic(
        nSensors,
        SensorSpacing,
        CurvatureRadius,
        pFocalCenter,
        RotationAngle=0):

    phiSensors = (
        (np.arange(nSensors) - (nSensors - 1)/2)
        * 2 * np.arctan(SensorSpacing/(2*CurvatureRadius))
    )

    pSensors = CurvatureRadius * np.vstack([
        -np.cos(phiSensors),
         np.sin(phiSensors)
    ])

    pSensors += pFocalCenter.reshape(2,1)

    theta = np.deg2rad(RotationAngle)

    R = np.array([
        [ np.cos(theta),  np.sin(theta)],
        [-np.sin(theta),  np.cos(theta)]
    ])

    pSensors = R @ pSensors

    return pSensors

# =====================================================
# SENSOR POSITIONS
# =====================================================

DSSs = get_sensor_positions_imasonic(
    SNo,
    SSpacing,
    CurRadius,
    Center,
    RotationAngle
)

# =====================================================
# INITIAL PARTICLE POSITIONS
# =====================================================

particlePos = DSSs.T.copy()

initialParticlePos = particlePos.copy()

# =====================================================
# PLOT SETUP
# =====================================================

plt.ion()

fig, ax = plt.subplots(figsize=(8,8))

ax.plot(
    DSSs[0,:]*1000,
    DSSs[1,:]*1000,
    'ro',
    markersize=8,
    label='Sensors'
)

ax.plot(
    birdPos[0]*1000,
    birdPos[1]*1000,
    'b*',
    markersize=18,
    label='Bird'
)

particlePlot = ax.scatter(
    particlePos[:,0]*1000,
    particlePos[:,1]*1000,
    s=50,
    c='black',
    label='Particles'
)

ax.set_xlabel("X Position (mm)")
ax.set_ylabel("Y Position (mm)")
ax.set_title("Particles Moving Toward Bird")

ax.grid(True)
ax.axis('equal')
ax.legend()

# =====================================================
# DRAW INITIAL TRAJECTORIES
# =====================================================

trajLines = []

for k in range(SNo):

    line, = ax.plot(
        [particlePos[k,0]*1000],
        [particlePos[k,1]*1000],
        'k:'
    )

    trajLines.append(line)

# =====================================================
# ANIMATION
# =====================================================

numSteps = 100

for step in range(numSteps + 1):

    alpha = step / numSteps

    particlePos = (
        (1-alpha)*initialParticlePos
        + alpha*birdPos
    )

    particlePlot.set_offsets(
        particlePos * 1000
    )

    for k in range(SNo):

        trajLines[k].set_data(
            [
                initialParticlePos[k,0]*1000,
                particlePos[k,0]*1000
            ],
            [
                initialParticlePos[k,1]*1000,
                particlePos[k,1]*1000
            ]
        )

    ax.set_title(
        f"Particles Converging To Bird ({round(alpha*100)}%)"
    )

    plt.draw()
    plt.pause(0.05)

# =====================================================
# FINAL MESSAGE
# =====================================================

print("All particles reached the bird position.")

plt.ioff()
plt.show()