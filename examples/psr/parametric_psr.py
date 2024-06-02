import numpy as np
import pandas as pd 
import cantera as ct
import matplotlib.pyplot as plt
from PyNetsmoke.reactor import *

'''In this example, we run parametric PSR simulations with a mixture
of methane and hydrogen, for several equivalence ratios and residence times, 
so that we can construct a 2-D operating map'''

# First of all, set path to kinetic mechanisms
kinfile     = "/Users/matteosavarese/Desktop/Dottorato/Kinetics/GRI3.0/gri30.CKI"
thermofile  = "/Users/matteosavarese/Desktop/Dottorato/Kinetics/GRI3.0/thermo.dat"
canteramech = 'gri30.cti' 

# Set path to OpenSMOKEpp installation folder
ospath = "/Users/matteosavarese/Desktop/Dottorato/OpenSmoke/opensmoke++suite-0.15.0.serial.osx"

# Create the gas object using cantera
gas = ct.Solution(canteramech)
gas.TP = 300.0, 101325.0

# Set equivalence ratio to one
fuel = 'CH4:0.5, H2:0.5'
oxi  = 'O2:0.21, N2:0.79'

# Setup parametric study
volume = 1.0    # If this is fixed and residence time is given, will adapt mass flowrate
# Set parametric study with tau
npt1 = 5
tau_span = np.array([0.001, 0.01, 0.1, 1.0, 10.0]) 
# Select also equivalence ratio range
npt2 = 21
phi_span = np.linspace(0.6, 1.2, npt2)

# Select variable to export, alternatively you can save output cantera mixtures in a list
NO_out = np.zeros((npt1, npt2))

# Start parametric analysis
for i in range(npt1):
    for j in range(npt2):
        # Set gas equivalence ratio
        gas.set_equivalence_ratio(phi_span[j], fuel=fuel, oxidizer=oxi) # by default basis is mol
        # Create the reactor object
        MyPSR = Reactor('PSR', isothermal=False, volume=volume, tau=tau_span[i], Mf=None, P=gas.P, 
                        InletMixture=gas, InitialStatus=gas, sp_threshold=1e-5,
                        CanteraMech=canteramech,
                        KinFile=kinfile, ThermoFile=thermofile, PreProcessor=True)
        # Write reactor input
        filepath = 'MyPSR'
        MyPSR.WriteInput(filepath=filepath)
        # Run simulation
        MyPSR.RunSimulation(ospath)
        # Extract output (cantera solution object)
        O = MyPSR.ExtractOutput(filepath=filepath)
        print(O.report())
        time.sleep(1)
        print("NO out = ", 1e6*O.X[gas.species_index('NO')], ' ppm')
        NO_out[i,j] = O.X[gas.species_index('NO')]

# Plot results
plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots(figsize=(6,4))
p1 = ax.plot(phi_span, NO_out[0,:], 'k-', label=str(tau_span[0])+' s')
p2 = ax.plot(phi_span, NO_out[1,:], 'b-.', label=str(tau_span[1])+' s')
p3 = ax.plot(phi_span, NO_out[0,:], 'r--', label=str(tau_span[0])+' s')
ax.legend()
ax.set_xlabel('$\phi$')
ax.set_ylabel('NO [mass frac]')
fig.tight_layout()
plt.show()





