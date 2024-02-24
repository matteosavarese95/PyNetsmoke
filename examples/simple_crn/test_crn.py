import numpy as np
import pandas as pd 
import cantera as ct
import matplotlib.pyplot as plt
from PyNetsmoke.reactor import *
from PyNetsmoke.reactor_network import *

### Single PSR simulation ###

# First of all, set path to kinetic mechanisms
kinfile     = "/Users/matteosavarese/Desktop/Dottorato/Kinetics/GRI3.0/gri30.CKI"
thermofile  = "/Users/matteosavarese/Desktop/Dottorato/Kinetics/GRI3.0/thermo.dat"
canteramech = 'gri30.cti' 

# Path to NetSMOKEpp.sh
netsmoke_path = "/Users/matteosavarese/Desktop/Dottorato/Github/NetSMOKEpp/SeReNetSMOKEpp-master/projects/Linux"

# Create inlet reactor
# Main reactor settings
rtype = 'PSR'
isothermal = False
tau = 1e-6
# Create a cantera onbject for the inlet mixture
gas = ct.Solution(canteramech)
gas.TP = 300.0, 101325.0
# Set equivalence ratio to one
fuel = 'CH4:1'
oxi  = 'O2:0.21, N2:0.79'
gas.set_equivalence_ratio(1.0, fuel=fuel, oxidizer=oxi) # by default basis is mol

### Create reactors 

# Create the reactor object
M0 = 1.0    # reference 1.0 kg/s
r0 = Reactor('PSR', isothermal=False, tau=tau, Mf=M0, P=gas.P, 
                 InletMixture=gas, InitialStatus=gas, sp_threshold=1e-5,
                 CanteraMech='gri30.cti',
                 isinput=True)

# Select recirculation
rec = 2.00
# Create flame reactor
M1 = M0*(1+rec)
r1 = Reactor('PSR', isothermal=False, tau=1.0, Mf=M1, P=gas.P, 
                 InletMixture=gas, InitialStatus=gas, sp_threshold=1e-5,
                 CanteraMech='gri30.cti',
                 isoutput=True)

### Create connection matrix ###
# element i-j is the mass going from reactor i to j
# inlet outlet are specified as inputs to reactors and should not be considered here
rconns = np.array([[0.0, M0],
                   [0.0, M1-M0]])

# Create reactors list
rlist = [r0, r1]

# Create reactor network object
rn = ReactorNetwork(rlist, rconns, kinfile, thermofile)
rn.WriteNetworkInput()

# Run simulation
rn.RunSimulation(netsmoke_path)

# Extract outputs 
Mlist = rn.ExtractOutputs()
print(Mlist[1].report())



