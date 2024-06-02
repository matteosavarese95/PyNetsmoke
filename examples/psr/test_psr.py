import numpy as np
import pandas as pd 
import cantera as ct
import matplotlib.pyplot as plt
from PyNetsmoke.reactor import *

### Single PSR simulation ###

# First of all, set path to kinetic mechanisms
kinfile     = "/Users/matteosavarese/Desktop/Dottorato/Kinetics/GRI3.0/gri30.CKI"
thermofile  = "/Users/matteosavarese/Desktop/Dottorato/Kinetics/GRI3.0/thermo.dat"
canteramech = 'gri30.cti' 

# Set path to OpenSMOKEpp installation folder
ospath = "/Users/matteosavarese/Desktop/Dottorato/OpenSmoke/opensmoke++suite-0.15.0.serial.osx"

# Main reactor settings
rtype = 'PSR'
isothermal = False
volume = 1.0
tau = 1.0

# Create a cantera onbject for the inlet mixture
gas = ct.Solution(canteramech)
gas.TP = 300.0, 101325.0
# Set equivalence ratio to one
fuel = 'CH4:1'
oxi  = 'O2:0.21, N2:0.79'
gas.set_equivalence_ratio(1.0, fuel=fuel, oxidizer=oxi) # by default basis is mol

# Create the reactor object
MyPSR = Reactor('PSR', isothermal=False, volume=volume, tau=tau, Mf=None, P=gas.P, 
                 InletMixture=gas, InitialStatus=gas, sp_threshold=1e-5,
                 CanteraMech='gri30.cti',
                 KinFile=kinfile, ThermoFile=thermofile, PreProcessor=True)

# Write reactor input
filepath = 'MyPSR'
MyPSR.WriteInput(filepath=filepath)

# Run simulation
MyPSR.RunSimulation(ospath)

# Extract output
M = MyPSR.ExtractOutput(filepath=filepath)
print(M.report())