import numpy as np
import cantera as ct
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time
from PyNetsmoke.reactor import *

### Create class for reactor networks
class ReactorNetwork:
    def __init__(self, Rlist, MassFlowrates, KinFile, ThermoFile,
                 MinIt=5, MaxIt=500, AtomicThreshold=1e-4, NonIsothermalThreshold=1e-4,
                 MaxUnbalance=0.05, SpeciesMonitor=['NO','CO'], CanteraMech='gri30.cti'):
        
        # Default initializer
        self.Rlist = Rlist                                          # List of reactors
        self.MassFlowrates = MassFlowrates                          # Matrix of internal mass flowrates
        self.KinFile = KinFile                                      # Path to kinetic file
        self.ThermoFile = ThermoFile                                # Path to thermo file
        self.MinIt = MinIt                                          # Minimum number of iterations
        self.MaxIt = MaxIt                                          # Maximum number of iterations
        self.AtomicThreshold = AtomicThreshold                      # Atomic error threshold
        self.NonIsothermalThreshold = NonIsothermalThreshold        # Non isothermal error threshold
        self.MaxUnbalance = MaxUnbalance                            # Max unbalance internal mass flowrates
        self.SpeciesMonitor = SpeciesMonitor                        # List of species to monitor
        self.CanteraMech = CanteraMech                              # Cantera chemical mechanism

        # Check that Rlist and MassFlowrates have same dimensions
        nr = len(Rlist)
        (nr1,nr2) = np.shape(MassFlowrates)

        if nr1 != nr2:
            raise ValueError('Mass flowrates matrix is non square. Check!')

        if nr != nr1:
            raise ValueError('Number of reactors not consistent between reactor list and mass flowrates')
        
        # Number of reactors
        self.Nr = nr1
        

    # Write input file function
    def WriteNetworkInput(self, FolderName='ReactorNetwork', DicName='input.dic'):

        '''This function will write only the global input.dic file for the 
        reactor network simulation. The other reactors, you must write their input file
        singluarly'''

        # Get current working directory
        cwd = os.getcwd()
        self.WorkingDir = cwd

        OutputFile = FolderName + '/' + DicName
        if os.path.exists(FolderName) == False:
            os.mkdir(FolderName)

        # Write single reactors inputs in FolderName
        idr = 0
        for r in self.Rlist:
            r.WriteInput(filepath=FolderName, Rid=idr)
            idr += 1

        # Identify PSRs and PFRs
        id_psrs = []; id_pfrs = []
        for i in range(len(self.Rlist)):
            if self.Rlist[i].Rtype == 'PSR':
                id_psrs.append(i)
            elif self.Rlist[i].Rtype == 'PFR':
                id_pfrs.append(i)

        # Number of reactors
        Nr = np.shape(self.MassFlowrates)[0]
        # Non zero connections
        Nconns = np.count_nonzero(self.MassFlowrates)

        # Count inlets
        InletList = []
        OutletList = []
        for i in range(Nr):
            if self.Rlist[i].isinput == True:
                InletList.append(i)
            elif self.Rlist[i].isoutput == True:
                OutletList.append(i)
        
        if len(InletList) == 0 or len(OutletList) == 0:
            raise ValueError('No inlets or outlets detected. Specify inlet/outlet reactors')

        # Write output file 
        with open(OutputFile, 'w') as f:
            f.write('Dictionary ReactorNetwork \n { \n')
            f.write('@KineticsPreProcessor     kinetic-mechanism;\n')
            f.write('@MinIterations                     %d;\n' % self.MinIt)
            f.write('@MaxIterations                     %d;\n' % self.MaxIt)
            f.write('@AtomicErrorThreshold              %e;\n' % self.AtomicThreshold)
            f.write('@NonIsothermalErrorThreshold       %e;\n' % self.NonIsothermalThreshold)
            f.write('@MaxUnbalance                      %e;\n' % self.MaxUnbalance)
            # Write perfectly stirred reactors
            if len(id_psrs) > 0:
                f.write('@PerfectlyStirredReactors \n')
                for i in range(len(id_psrs)):
                    if i < len(id_psrs)-1:
                        f.write('%d       input.cstr.%d.dic\n' % (id_psrs[i], id_psrs[i]))
                    else:
                        f.write('%d       input.cstr.%d.dic;\n' % (id_psrs[i], id_psrs[i]))
            # Write plug flow reactors
            if len(id_pfrs) > 0:
                f.write('@PlugFlowReactors \n')
                for i in range(len(id_pfrs)):
                    if i < len(id_pfrs)-1:
                        f.write('%d       input.pfr.%d.dic\n' % (id_pfrs[i], id_pfrs[i]))
                    else:
                        f.write('%d       input.pfr.%d.dic;\n' % (id_pfrs[i], id_pfrs[i]))
            # Write internal connections
            ncounts = 0  # Connections counter
            f.write('@InternalConnections     ')
            for i in range(Nr):
                for j in range(Nr):
                    if self.MassFlowrates[i,j] != 0 and ncounts < Nconns - 1:
                        f.write('%d     %d     %e \n' % (i, j, self.MassFlowrates[i,j]))
                        ncounts += 1
                    elif self.MassFlowrates[i,j] != 0 and ncounts == Nconns - 1:
                        f.write('%d     %d     %e;\n' % (i, j, self.MassFlowrates[i,j]))
                        ncounts += 1
            # Write inlet streams
            f.write('@InputStreams     ')
            for i in range(len(InletList)):
                if i < len(InletList) - 1:
                    f.write('%d     %e \n' % (InletList[i], self.Rlist[InletList[i]].Mf))
                elif i == len(InletList) - 1:
                    f.write('%d     %e; \n' % (InletList[i], self.Rlist[InletList[i]].Mf))
            # Write outlet streams
            f.write('@OutputStreams     ')
            for i in range(len(OutletList)):
                ri = OutletList[i]
                mi = np.abs(np.sum(self.MassFlowrates[ri,:]) - np.sum(self.MassFlowrates[:,ri]))
                if i < len(OutletList) - 1:
                    f.write('%d     %e \n' % (ri, mi))
                elif i == len(OutletList) - 1:
                    f.write('%d     %e; \n' % (ri, mi))
            # Write species to monitor
            f.write('@SpeciesToMonitor     ')
            for i in range(len(self.SpeciesMonitor)):
                f.write('%s\t' % self.SpeciesMonitor[i])
            f.write(';\n')
            # Verbosity Level
            f.write('@VerbosityLevel     1;\n')
            f.write('} \n \n')

            # Kinetic mechanism dictionary
            f.write('Dictionary kinetic-mechanism \n { \n')
            f.write('@Kinetics          %s;\n' % self.KinFile)
            f.write('@Thermodynamics    %s;\n' % self.ThermoFile)
            f.write('@Output     kinetics;\n')

            f.write('} \n')

        # Return simulation folder
        self.SimFolder = FolderName
        return self
        
    # Command to run simulation
    def RunSimulation(self, netsmoke_path):

        # Check if SimFolder exists
        if os.path.exists(self.SimFolder) == False:
            raise ValueError('Simulation folder is not specified as attribute in reactor network class')
        
        # Create Run.sh file for simulation running
        runfile = self.SimFolder + '/' + 'Run.sh'
        with open(runfile, 'w') as f:
            f.write(netsmoke_path+'/SeReNetSMOKEpp.sh --input input.dic')
        
        # Create string for command to be executed
        os.chdir(self.SimFolder)
        command = 'sh Run.sh'
        os.system(command)
        # Return to working directory
        os.chdir(self.WorkingDir)

    def ExtractSingleOutput(self, Rid):

        '''This function will extract the output of the specific
        reactor Reactor.Rid that is specified and return it as a 
        cantera quantity object'''

        # Check if SimFolder exists
        if os.path.exists(self.SimFolder) == False:
            raise ValueError('Simulation folder is not specified as attribute in reactor network class')
        
        # Extract data for reactor Rid
        filename = self.SimFolder + '/Output/Reactor.' + str(Rid) + '/Output.out'
        df = pd.read_csv(filename, sep='\s+')

        # Export T, P
        T = df.values[-1,4]
        P = df.values[-1,5]

        # In columns, locate all the strings that have '_x" inside
        sp_list = []
        X_val   = []
        for i, col in enumerate(df.columns):
            ss = col.split('_x')
            if len(ss) > 1:
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
        # Create quantity object with mass from reactor
        M = ct.Quantity(gas, constant='HP', mass=self.Rlist[Rid].Mf)

        return M
    
    def ExtractOutputs(self):

        '''This function will extract the output of the specific
        reactor Reactor.Rid that is specified and return it as a 
        cantera quantity object'''

        # Check if SimFolder exists
        if os.path.exists(self.SimFolder) == False:
            raise ValueError('Simulation folder is not specified as attribute in reactor network class')
        
        # Initialize list of cantera quantities
        Mlist = []
        
        # Extract data for reactor Rid
        for i in range(self.Nr):
            M = self.ExtractSingleOutput(i)
            Mlist.append(M)
    
        return Mlist
    
    def ExtractSingleInput(self, Rid):

        '''This function will extract the input of the specific
        reactor Reactor.Rid that is specified and return it as a 
        cantera quantity object'''

        # Check if SimFolder exists
        if os.path.exists(self.SimFolder) == False:
            raise ValueError('Simulation folder is not specified as attribute in reactor network class')
        
        # Extract data for reactor Rid
        filename = self.SimFolder + '/Output/Reactor.' + str(Rid) + '/log.inlet'
        df = pd.read_csv(filename, sep='\s+')

        # Export T, P
        T = df.values[-1,1]

        # Extract data for reactor Rid
        filename2 = self.SimFolder + '/Output/Reactor.' + str(Rid) + '/Output.out'
        df2 = pd.read_csv(filename2, sep='\s+')
        P = df2.values[-1,5]

        # In columns, locate all the strings that have '_x" inside
        sp_list = []
        Y_val   = []
        for i, col in enumerate(df.columns[2:]):
            ss = col.split('(')
            if len(ss) > 1:
                sp_list.append(ss[0])
                Y_val.append(df.values[-1,i])

        # Create solution
        gas = ct.Solution(self.CanteraMech)
        # Extract number of species
        ns = gas.n_species
        sp_list = gas.species_names
        Y_old = gas.Y
        Y_new = Y_old
        for i, sp in enumerate(sp_list):
            Y_new[gas.species_index(sp)] = Y_val[i]

        # Set new gas state
        gas.TPY = T, P, Y_new
        # Create quantity object with mass from reactor
        M = ct.Quantity(gas, constant='HP', mass=self.Rlist[Rid].Mf)

        return M
    
    def ExtractInputs(self):

        '''This function will extract the inputs of the specific
        reactor Reactor.Rid that is specified and return it as a 
        cantera quantity object'''

        # Check if SimFolder exists
        if os.path.exists(self.SimFolder) == False:
            raise ValueError('Simulation folder is not specified as attribute in reactor network class')
        
        # Initialize list of cantera quantities
        Mlist = []
        
        # Extract data for reactor Rid
        for i in range(self.Nr):
            M = self.ExtractSingleInput(i)
            Mlist.append(M)
    
        return Mlist
    
    def ExtractOutputSingleVar(self, varname):

        '''This function will extract only one single variable. It will be 
        faster because we do not need to create the Cantera object representing the phase'''

        # Check if SimFolder exists
        if os.path.exists(self.SimFolder) == False:
            raise ValueError('Simulation folder is not specified as attribute in reactor network class')
        
        # Find the desired variable
        if varname == "T":
            varid = 4
        elif varname == "time" or varname == "tau":
            varid = 0
        else:
            f0 = self.SimFolder + '/Output/Reactor.0/Output.out'
            df = pd.read_csv(f0, sep='\s+')
            for j, col in enumerate(df.columns):
                ss = col.split('_x')
                if len(ss) > 0:
                    if varname == ss[0]:
                        varid = j
            if varid == 0:
                raise ValueError("The variable was not found in columns!")

        # Initialize output
        yout = np.zeros(self.Nr)
        for i in range(self.Nr):    
            filename = self.SimFolder + '/Output/Reactor.' + str(i) + '/Output.out'
            df = pd.read_csv(filename, sep='\s+')
            yout[i] = df.values[-1,varid]

        return yout
    

'''Class for generalized CRN'''
class GeneralCRN:

    def __init__(self, KinFile, Thermofile):

        self.Kinfile_ = KinFile
        self.Thermofile_ = Thermofile

    # Define function to initialize input layer
    def InputLayer(self, Inputs):

        # Number of inputs
        nin = len(Inputs)

        # Create each node as input node
        nodes = []
        locs  = []
        for i in range(nin):
            nodes.append(("I"+str(i),  "PSR"))
            locs.append((1, i))

        # Create the attribute of layers
        self.layers_ = {"input":nodes}
        self.node_locations_ = locs
        # Initialize connections
        self.connections_ = []
        self.splitting_ratios_ = []

        return self
    
    def AddConnectedLayer(self, nr, rtype="PSR"):

        # Count existing nodes
        total_nodes = sum(len(nodes) for nodes in self.layers_.values())

        # Create new connections with all the nodes in the previous layer
        n_last = len(list(self.layers_.values())[-1])
        last_nodes = list(self.layers_.values())[-1]
        last_layer_name = list(self.layers_.keys())[-1]
        # Check if last_layer_name is input
        if last_layer_name == "input":
            ns = "H0"
            nl = 0
            layername = 'hidden0'
        else:
            nl = int(last_layer_name[6:]) + 1
            ns = "H" + str(nl)
            layername = 'hidden'+str(nl)

        newconns = []
        splitting_ratios = []
        split_ratio = round(1/n_last, 4)
        for i in range(n_last):
            for j in range(nr):
                
                # Name of the new node
                newnode_name = ns + str(j)
                
                # Add new connection between previous layer and new layer
                newconns.append((last_nodes[i][0], newnode_name))
                splitting_ratios.append(split_ratio)

        # Create new hidden layer in the layers
        hidden_layer = []
        newlocs = []
        for i in range(nr):
            newnode_name = ns + str(i)
            hidden_layer.append((newnode_name, rtype))
            newlocs.append((int(nl+1)*2+1, i))

        # Update layers
        self.layers_[layername] = hidden_layer
        # Update nodes locations
        for item in newlocs:
            self.node_locations_.append(item)
        # Update nodes connections
        for item in newconns:
            self.connections_.append(item)
        # Update splitting rations
        for item in splitting_ratios:
            self.splitting_ratios_.append(item)

        return self
    
    def AddOutputLayer(self, nr, rtype="PSR"):

        # Count existing nodes
        total_nodes = sum(len(nodes) for nodes in self.layers_.values())

        # Create new connections with all the nodes in the previous layer
        n_last = len(list(self.layers_.values())[-1])
        last_nodes = list(self.layers_.values())[-1]
        last_layer_name = list(self.layers_.keys())[-1]
        nl = int(last_layer_name[6:]) + 1
        ns = "O"
        layername = 'output'

        newconns = []
        splitting_ratios = []
        split_ratio = round(1/n_last, 4)
        for i in range(n_last):
            for j in range(nr):
                
                # Name of the new node
                newnode_name = ns + str(j)
                
                # Add new connection between previous layer and new layer
                newconns.append((last_nodes[i][0], newnode_name))
                splitting_ratios.append(split_ratio)

        # Create new hidden layer in the layers
        hidden_layer = []
        newlocs = []
        for i in range(nr):
            newnode_name = ns + str(i)
            hidden_layer.append((newnode_name, rtype))
            newlocs.append((int(nl+1)*2+1, i))

        # Update layers
        self.layers_[layername] = hidden_layer
        # Update nodes locations
        for item in newlocs:
            self.node_locations_.append(item)
        # Update nodes connections
        for item in newconns:
            self.connections_.append(item)
        # Update splitting rations
        for item in splitting_ratios:
            self.splitting_ratios_.append(item)

        return self
    
    def CreateGraph(self, plot=False):
        '''This function creates the graph of the CRN using the networkX library'''
        G = nx.Graph()
        # Add the nodes
        for layer, nodes in self.layers_.items():
            for node, node_type in nodes:
                if node_type == "PFR":
                    shape="box"
                    G.add_node(node, type=node_type, shape=shape)
                else:
                    G.add_node(node, type=node_type)

        # Add the connections
        for i in range(len(self.connections_)):
            G.add_edge(self.connections_[i][0], self.connections_[i][1], weight=float(self.splitting_ratios_[i]))

        # Plot if true
        if plot == True:
            # Create dictionary of nodes positions
            locations = {}
            nodes     = list(G.nodes())
            for i in range(len(nodes)):
                locations[nodes[i]] = self.node_locations_[i]
            node_colors = {'PSR': 'lightblue', 'PFR': 'lightgreen'}
            node_type = nx.get_node_attributes(G, 'type')
            nx.draw(G, pos=locations, with_labels=True, node_color=[node_colors[node_type[node]] for node in G.nodes()],
                node_size=1500, edge_color='gray')
            plt.show()

        return G


        









        





