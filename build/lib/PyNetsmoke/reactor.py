import numpy as np
import cantera as ct
import os
import pandas as pd
import time

class Reactor:
    def __init__(self, Rtype, isothermal=False, volume=None, tau=None, Mf=None, P=None, 
                 L=None, D=None, Ac=None, Aq=None, Uq=None, Tenv=None, AcOverP=None,
                 InletMixture=None, InitialStatus=None, sp_threshold=1e-5,
                 isinput=False, isoutput=False,
                 CanteraMech='gri30.cti', KinPath="dummy",
                 KinFile=None, ThermoFile=None, PreProcessor=False):
        
        # Check if Rtype exists
        if Rtype != 'PSR' and Rtype != 'PFR':
            raise ValueError('Unknown reactor type. Specify PSR or PFR (must be specified upper case)')
        
        self.Rtype = Rtype                          # Reactor type ('PSR or PFR)
        self.isothermal = isothermal                # Isothermal or Non-isothermal 
        # Reactor parameters
        self.volume = volume                        # Reactor volume            [m3]
        self.tau = tau                              # Reactor residence time    [s]
        self.Mf = Mf                                # Mass flowrate             [kg/s]
        self.P = P                                  # Pressure                  [Pa]
        self.L = L                                  # Length                    [m]
        self.D = D                                  # Diameter                  [m]
        self.Ac = Ac                                # Cross sectional area      [m2]
        self.Aq = Aq                                # Heat exchange area        [m2]
        self.Uq = Uq                                # Heat transfer coefficient [W/m2/K]
        self.Tenv = Tenv                            # Environment temperature   [K]
        self.AcOverP = AcOverP                      # Cross section over per.   [m]
        # Reactor initialization and inlets
        self.InletMixture = InletMixture            # Inlet mixture (cantera object)
        self.InitialStatus = InitialStatus          # InitialStatus (cantera object)
        self.sp_threshold = sp_threshold            # Threshold to write initial composition dictionary
        self.CanteraMech = CanteraMech              # Chemical mechanism file cantera
        # Only for reactor networks
        self.isinput = isinput                      # True if is an inlet reactor
        self.isoutput = isoutput                    # True if is an outlet reactor
        # ROPA options
        self.ropa = False                           # True when you do SetROPA
        self.RopaThreshold = 0.001                  # Relative threshold for reaction rate
        self.RopaSpecies = None                     # ROPA species
        self.RopaThreshold = None                   # Threshold for ROPA species
        # Kinetic mechanism informations
        self.KinPath = KinPath                      # Kinetics Folder (to pre-processed kinetic mechanism)
        self.KinFile = KinFile                      # Path to kinetic mechanism file (usually .dat, .kin...)
        self.ThermoFile = ThermoFile                # Path to thermodynamic file
        self.PreProcessor = PreProcessor            # If kinetic and thermo file are specified, it must be true

        # Initialize file path
        self.FilePath = None


    def SetROPA(self, species, reference, threshold=0.001):
        '''This additional function is used to setup a ROPA analysis. Check
        out possible options and their meaning in the OpenSMOKEpp documentation'''

        # Activate flag
        self.ropa = True
        self.RopaSpecies = species
        self.ReferenceSpecies = reference
        self.RopaThreshold = threshold
        return self

    def WriteInput(self, filepath=None, Rid=None):

        ''' This function will write the reactor input dictionary
        file. If Rid is specified, the input file name will be coherent
        with the number of the reactor in the reactor network'''

        # Check if Rid was specified
        if Rid != None:
            self.Rid = Rid
        else:
            self.Rid = None
        # Check if filepath was specified, otherwise save it where you are
        if filepath != None:
            self.FilePath = filepath
        else:
            self.FilePath = '.'
            
        # PSR input dictionary
        if self.Rtype == 'PSR':
            if filepath == None and Rid == None:
                filename = 'input.dic'
            # If Rid was specified, change name to input file
            elif filepath == None and isinstance(Rid, int):
                filename = 'input.cstr.' + str(Rid) + '.dic'
            # If also the path was specified save it there
            elif isinstance(filepath, str) and isinstance(Rid, int):
                filename = filepath + '/input.cstr.' + str(Rid) + '.dic'
            # If path was specified but not Rid, save it there just as input.dic
            elif isinstance(filepath, str) and Rid == None:
                if os.path.exists(filepath) == False:
                    os.mkdir(filepath)
                filename = filepath + '/input.dic'

        # PFR input dictionary (same rules as before)
        elif self.Rtype == 'PFR':
            if filepath == None and Rid == None:
                filename = 'input.dic'
            elif filepath == None and isinstance(Rid, int):
                filename = 'input.pfr.' + str(Rid) + '.dic'
            elif isinstance(filepath, str) and isinstance(Rid, int):
                filename = filepath + '/input.pfr.' + str(Rid) + '.dic'
            elif isinstance(filepath, str) and Rid == None:
                if os.path.exists(filepath) == False:
                    os.mkdir(filepath)
                filename = filepath + '/input.dic'

        # Open file
        with open(filename, 'w') as f:
            # Check if it is a PSR or a PFR
            if self.Rtype == 'PSR':
                f.write('Dictionary PerfectlyStirredReactor \n { \n')
            elif self.Rtype == 'PFR':
                f.write('Dictionary PlugFlowReactor \n { \n')

            # Write attributes
                
            # Options for PSR
            if self.isothermal == True and self.Rtype == 'PSR':
                f.write('     @Type     Isothermal-ConstantPressure;\n')
            elif self.isothermal == False and self.Rtype == 'PSR':
                f.write('     @Type     NonIsothermal-ConstantPressure;\n')
            # Options for PFR
            elif self.isothermal == True and self.Rtype == 'PFR':
                f.write('     @Type     Isothermal;\n')
                f.write('     @ConstantPressure   true;\n')
            elif self.isothermal == False and self.Rtype == 'PFR':
                f.write('     @Type     NonIsothermal;\n')
                f.write('     @ConstantPressure   true;\n')

            # Kinetic dictionary
            if self.PreProcessor == False:
                f.write('     @KineticsFolder     %s;\n' %self.KinPath)
            elif self.PreProcessor == True:
                f.write('      @KineticsPreProcessor     kinetic-mechanism; \n')
                if self.KinFile == None or self.ThermoFile == None:
                    raise ValueError('Preprocessor is selected but path to thermo or kinetic mech is not specified!')

            # Value instances
            # Volume
            if isinstance(self.volume, (float, int)):
                f.write('     @Volume    %e m3;\n' %self.volume)
            # Residence time
            if isinstance(self.tau,  (float, int)):
                f.write('     @ResidenceTime    %e s;\n' %self.tau)
            # Mass flowrate
            if isinstance(self.Mf,  (float, int)):
                f.write('     @MassFlowRate    %e kg/s;\n' %self.Mf)
            # Length
            if isinstance(self.L,  (float, int)):
                f.write('     @Length    %e m;\n' %self.L)           
            # Diameter
            if isinstance(self.D,  (float, int)):
                f.write('     @Diameter    %e m;\n' %self.D)
            # Heat exchange area
            if isinstance(self.Aq,  (float, int)):
                f.write('     @ExchangeArea    %e m2;\n' %self.Aq)        
            # Heat transfer coefficient
            if isinstance(self.Uq,  (float, int)):
                f.write('     @GlobalThermalExchangeCoefficient    %e W/m2/K;\n' %self.Uq) 
            # Environment temperature
            if isinstance(self.Tenv,  (float, int)):
                f.write('     @EnvironmentTemperature    %d K;\n' %self.Tenv)
            # Cross section over perimeter
            if isinstance(self.AcOverP,  (float, int)):
                f.write('     @CrossSectionOverPerimeter    %d m;\n' %self.AcOverP)

            # Initial status
            if self.InitialStatus != None:
                f.write('     @InitialStatus     initial-status;\n')
            
            # Inlet status
            if self.InletMixture != None:
                f.write('     @InletStatus     inlet-mixture;\n')

            # ROPA option
            if self.ropa == True:
                f.write('     @OnTheFlyROPA      ropa;\n')

            # Close dictionary
            f.write('} \n')

            ### Additional dictionaries ####
            if self.InletMixture != None:
                M = self.InletMixture   # Cantera object
                # Extract main quantities
                T = M.T
                P = M.P
                # Extract composition
                X = M.X
                # Extract names
                names = M.species_names
                # Select only a small portion of species to be written
                mfsp = []
                spls = []
                thresh = 1e-6
                for i in range(len(names)):
                    if X[i] > thresh:
                        mfsp.append(X[i])
                        spls.append(names[i])
                f.write('Dictionary inlet-mixture \n { \n')
                f.write('     @Temperature     %e K;\n' % T)
                f.write('     @Pressure        %e Pa;\n' % P)
                f.write('@Moles     ')
                # Write species
                for i in range(len(mfsp)):
                    if i < len(mfsp)-1:
                        f.write('%s %f \n' % (spls[i], mfsp[i]))
                    else:
                        f.write('%s %f; \n' % (spls[i], mfsp[i]))
                f.write('} \n')

            ### Additional dictionaries ####
            if self.InitialStatus != None:
                M = self.InletMixture   # Cantera object
                # Extract main quantities
                T = M.T
                P = M.P
                # Extract composition
                X = M.X
                # Extract names
                names = M.species_names
                # Select only a small portion of species to be written
                mfsp = []
                spls = []
                thresh = 1e-6
                for i in range(len(names)):
                    if X[i] > thresh:
                        mfsp.append(X[i])
                        spls.append(names[i])
                f.write('Dictionary initial-status \n { \n')
                f.write('     @Temperature     %e K;\n' % 2500.0)
                f.write('     @Pressure        %e Pa;\n' % P)
                f.write('     @Moles     ')
                # Write species
                for i in range(len(mfsp)):
                    if i < len(mfsp)-1:
                        f.write('%s %f \n' % (spls[i], mfsp[i]))
                    else:
                        f.write('%s %f; \n' % (spls[i], mfsp[i]))
                f.write('} \n')

            if self.ropa == True:
                f.write('Dictionary ropa \n{ \n')
                f.write('     @Species ')
                for i in range(len(self.RopaSpecies)):
                    f.write(' %s' % self.RopaSpecies[i])
                f.write(';\n')
                # Reference species
                f.write('     @ReferenceSpecies ')
                for i in range(len(self.ReferenceSpecies)):
                    f.write(' %s' % self.ReferenceSpecies[i])
                f.write(';\n')
                # Threshold
                f.write('     @Threshold    %e;\n' % self.RopaThreshold)
                f.write('} \n')

            # Kinetic mechanism dictionary
            if self.PreProcessor == True:
                f.write('\nDictionary kinetic-mechanism \n{ \n')
                f.write('     @Kinetics     %s; \n' % self.KinFile)
                f.write('     @Thermodynamics    %s;   \n' % self.ThermoFile)
                f.write('     @Output     kinetics; \n } \n')

        return self
    
    def RunSimulation(self, ospath):

        '''This function will run the simulation of the single reactor. 
        Therefore, you must have installed the executable files 
        PerfectlyStirredReactor.sh and PlugFlowReactor.sh in the 
        opensmoke++suite-0.vv.0.serial.osx folder.'''

        if os.path.exists(ospath) == False:
            raise ValueError("Your specified path %s does not exists!" % ospath)

        # Generate the command
        if self.Rtype == 'PSR':
            command = 'export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:' + ospath +'/lib ; ' + ospath + '/bin/OpenSMOKEpp_PerfectlyStirredReactor.sh --input input.dic'
        elif  self.Rtype == 'PFR':
            command = 'export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:' + ospath +'/lib ; ' + ospath + ' /bin/OpenSMOKEpp_PlugFlowReactor.sh --input input.dic'

        # Execute command
        if self.FilePath != None:
            os.chdir(self.FilePath)
            os.system(command)
            os.chdir('../')
        else:
            os.system(command)

        return self

    def ExtractRopa(self, filepath=None):

        '''Function to extract the ROPA output as an array of net reaction rates'''

        if filepath == None and self.Rid != None:
            filepath = 'ReactorNetwork/Output/Reactor.' + str(self.Rid) + '/ROPA.out'
        else:
            filepath = self.FilePath + '/Output/ROPA.out'

        net_rates = []
        for sp in self.RopaSpecies:
            with open(filepath, 'r') as f:
                spcounter = 0
                for line in f:
                    ss = line.split()
                    if len(ss) > 0:
                        if ss[0] == sp:
                            spcounter += 1
                            if spcounter == 2:
                                net_rates.append(float(ss[6]))

        return net_rates
    
    def ExtractInletComp(self, filepath=None, sp_threshold=1e-10):

        '''Function to extract the inlet status of the mixture to a certain
        reactor. This is done ONLY for reactor network reactors.'''

        if self.Rid == None:
            raise ValueError("This function is only for reactors in reactor networks!")

        if filepath == None:
            filepath = 'ReactorNetwork/Output/Reactor.' + str(self.Rid) + '/log.inlet'
        
        # Extract file
        df = pd.read_csv(filepath, sep='\s+')

        # Export T, P
        T = df.values[-1,1]
        P = 101325.0

        # In columns, locate all the strings that have '_x" inside
        sp_list = []
        Y_val   = []
        for i, col in enumerate(df.columns[2:], 2):
            ss = col.split('(')
            if len(ss) > 1:
                if df.values[-1,i] < 1:
                    sp_list.append(ss[0])
                    Y_val.append(df.values[-1,i])

        # Create solution
        gas = ct.Solution(self.CanteraMech)
        # Extract number of species
        ns = gas.n_species
        # sp_list = gas.species_names
        Y_old = gas.Y
        Y_new = Y_old
        for i, sp in enumerate(sp_list):
            if Y_val[i] > sp_threshold:
                Y_new[gas.species_index(sp)] = Y_val[i]
            else:
                Y_new[i] = 0.0

        # Set new gas state
        gas.TPY = T, P, Y_new

        return gas
    
    def ExtractOutput(self, filepath=None):

        '''This function will extract the output of a certain reactor and it will
        return it as a Cantera solution object.'''

        if filepath == None and self.Rid != None:
            filepath = 'ReactorNetwork/Output/Reactor.' + str(self.Rid) + '/Output.out'
        else:
            filepath = self.FilePath + '/Output/Output.out'

        df = pd.read_csv(filepath, sep='\s+')

        # Export T, P
        T = df.values[-1,4]
        P = df.values[-1,5]

        # In columns, locate all the strings that have '_x" inside
        sp_list = []
        X_val   = []
        for i, col in enumerate(df.columns):
            ss = col.split('(')
            if len(ss) > 1 and ss[0] != "rho[kg/m3]":
                if df.values[-1,i] < 1:
                    sp_list.append(ss[0])
                    X_val.append(df.values[-1,i])

        # Create solution
        gas = ct.Solution(self.CanteraMech)
        # Extract number of species
        ns = gas.n_species
        sp_list = gas.species_names
        X_old = gas.X
        X_new = X_old
        for i, sp in enumerate(sp_list):
            X_new[gas.species_index(sp)] = X_val[i]

        # Set new gas state
        gas.TPX = T, P, X_new

        return gas