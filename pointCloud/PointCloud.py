import os 
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import axes3d, Axes3D 

import diffpy.Structure 
import CifFile
from scipy.spatial import distance
from scipy.sparse import *
import networkx as nx

class PointCloud(object):
    def __init__(self, cifFilePath, extendedCloud = False, netInput = False):
        self.filePath = cifFilePath
        self.fileName = cifFilePath.split('/')[-1] # tail of filepath for name
        self.cifFile = CifFile.ReadCif(self.filePath) # Read cif in as a dictionary
        # self.molecule = diffpy.Structure.loadStructure(self.filePath) # store cif as a diffpy.Structure object
        self.lattice = self._generateLattice(self.cifFile) # self.molecule.lattice   # cell lengths and angles
        self.fractionalCoords = self._generateCoords(self.cifFile) # diffpy is wrong siomehow np.array(self.molecule.xyz)
        self.transformationMatrix = self._transformationMatrix(self.lattice)
        self.cartesianCoords = self.fractionalCoords.dot(self.transformationMatrix)
        self.reducedCloudFrac, self.reducedCloudCart = self._generateReducedCloud(self.fractionalCoords)
        
        if extendedCloud:           # Generate an extended cell, off by default
            self.generateExtendedCloud(self.lattice)
            
        if netInput:        # Generate an input for the neural net with default values (unit cell 1000 divisions)
            self.generateNetInput(self.cartesionCoords)
        
    def loadCifFile(self, cifFilePath):
        self.filePath = cifFilePath
        self.molecule = diffpy.Structure.loadStructure(self.filePath)
        self.cartesianCoords = np.array(self.molecule.xyz_cartn)
    
    def _generateLattice(self, cf):
        keys = list(cf.keys())
        keys.remove('global') # Just take the data block
        lattice = {}
        lattice['a'] = float(cf[keys[0]]['_cell_length_a'])
        lattice['b'] = float(cf[keys[0]]['_cell_length_b'])
        lattice['c'] = float(cf[keys[0]]['_cell_length_c'])
        lattice['alpha'] = float(cf[keys[0]]['_cell_volume'])
        lattice['beta'] = float(cf[keys[0]]['_cell_angle_beta'])
        lattice['gamma'] = float(cf[keys[0]]['_cell_angle_gamma'])
        lattice['volume'] = float(cf[keys[0]]['_cell_angle_gamma'])
        return lattice

    def _generateCoords(self, cf):
        keys = list(cf.keys())
        keys.remove('global') # Just take the data block
        xs = np.array(cf[keys[0]]['_atom_site_fract_x'], np.float32)
        ys = np.array(cf[keys[0]]['_atom_site_fract_y'], np.float32)        
        zs = np.array(cf[keys[0]]['_atom_site_fract_z'], np.float32)
        fractionalCoords = np.stack((xs, ys, zs), axis = 1)
        return fractionalCoords
        
    def _transformationMatrix(self, lattice):
        transformationMatrix = np.zeros((3,3))
        transformationMatrix[0][0] = lattice['a']
        transformationMatrix[1][0] = lattice['b'] * np.cos(np.pi * lattice['gamma'] / 180)
        transformationMatrix[2][0] = lattice['c'] * np.cos(np.pi * lattice['beta'] / 180)
        transformationMatrix[1][1] = lattice['b'] * np.sin(np.pi * lattice['gamma'] / 180)
        transformationMatrix[2][1] = lattice['c'] * (np.cos(np.pi * lattice['alpha'] / 180) - np.cos(np.pi * lattice['beta'] / 180) * np.cos(np.pi * lattice['gamma'] / 180) / np.sin(np.pi * lattice['gamma'] / 180))
        transformationMatrix[2][2] = lattice['volume'] / (lattice['a'] * lattice['b'] * np.sin(np.pi * lattice['gamma'] / 180))
        return transformationMatrix
    
    def _generateReducedCloud(self, pointCloud):
        # As there are 41 atoms in a single T2 molecule we use this to find the centre of each molecule
        self.numMols = pointCloud.shape[0] / 41
        print(self.numMols)
        reducedCloudFrac = np.zeros((self.numMols, 3)) # create empty matrix for coords
        for i in range(self.numMols):
            reducedCloudFrac[i] = (pointCloud[(41 * i) + 9] + pointCloud[(41 * i) + 22]) * 0.5
        return reducedCloudFrac, reducedCloudFrac.dot(self.transformationMatrix)
        
    def generateExtendedCloud(self, lattice, coords = None, reducedCloud = True):
        '''
        Extend the cloud in 7 directions, due to the mirror symmetry this is sufficient as we will be removing duplicate values
        
        Once extended calculate the adjacency matrix for the entire cloud and remove unnecessary repeated values
        '''
        
        if coords is None and reducedCloud is True:
            coords = self.reducedCloudFrac
        else:
            coords = self.cartesianCoords
            
        fractionalAxis = np.array([[lattice['a'], 0, 0], [0, lattice['b'], 0], [0, 0, lattice['c']]])
        cartesianAxis = fractionalAxis.dot(self.transformationMatrix)
        
        extendedCloud = np.concatenate((coords,     # Expand across all axis
                                       coords + cartesianAxis[0],
                                       coords + cartesianAxis[1],
                                       coords + cartesianAxis[2],
                                       coords + cartesianAxis[0] + cartesianAxis[1],
                                       coords + cartesianAxis[0] + cartesianAxis[2],
                                       coords + cartesianAxis[1] + cartesianAxis[2],
                                       coords + cartesianAxis[0] + cartesianAxis[1] + cartesianAxis[2]),
                                       axis = 0)
        self.extendedCloud = extendedCloud # save as a property

        adjacencyMatrix = self.generateAdjacencyMatrixNN(coords = extendedCloud, knn = extendedCloud.shape[0]) # Full adjacency matrix of cloud
        adjacencyMatrix[self.numMols:, :] = 0 # replace all distances outside the first unit cell with zero
        simplifiedAdjacencyMatrix = np.triu(adjacencyMatrix[:self.numMols, :]) # remove the inner cell repeated adjacencies (upper triangle only)
        flattenedDistances = np.sort(simplifiedAdjacencyMatrix.flatten()) # vector of distances from initial cell
        self.simplifiedMinimumDist = flattenedDistances[np.nonzero(flattenedDistances)] # take non zero elements and save
        
    def generateNetInput(self, coords, divisions = 1000, writeToFile = False, outputFilePath = '/netInputs/'):
        # netInput = np.zeros(divisions, divisions, divisions, dtype = int) # empty array for the net input
        normalizedCoords = np.zeros(coords.shape, dtype = int) # empty array for normalised coords
        xmax, xmin = coords.max(), coords.min()
        normalizedCoords = np.array(((coords - xmin) / (xmax - xmin) * divisions), dtype = int) # normalize coords
        indexingArray = np.zeros(normalizedCoords.shape[0], dtype=int)
        for i, row in enumerate(normalizedCoords):
            indexingArray[i] = row[0] + (row[1] * divisions) + (row[2] * divisions * divisions)
        
        self.sortedIndexArray = np.sort(indexingArray) # which input neurons we should fire
        
        if writeToFile:
            f = open(self.fileName + 'net', 'wb')
            currentIndex = 0
            for i in range(divisions):
                for j in range(divisions):
                    for k in range(divisions):
                        if currentIndex in self.sortedIndexArray:
                            f.write('1 ')   # write a one for present atoms
                        else:
                            f.write('0 ')   # else write a zero
                print(i)
            f.close()

    def sumAtomicFingerPrint(self, k = 12):
        print("adsfn")
        
    def atomicFingerPrint(self, adjacencyMatrixRow, k=12):
        #TODO FINISH THIS
        # Create adjacency matrix for entire cloud
        if self.adjacencyMatrix is None:
            # Take the first point in the file
            neighbourDistances = self.createAdjacencyMatrix(knn = self.cartesianCoords.shape[0])[0]
        else:
            # Take in the given row
            neighbourDistances = adjacencyMatrixRow
            
        # Remove our zero value for the atom in question
        indexOfRoot = np.argmin(neighbourDistances) # find index (TODO Remove?)
        neighbourDistances = np.delete(neighbourDistances, indexOfRoot)  # remove
        
        # Take the minimum distance Ri0
        Rmin = np.amin(neighbourDistances)
        Rmax = Rmin * 6
        
        # Remove all distances greater than Rmax
        neighbourDistances = neighbourDistances[neighbourDistances < Rmax] 
        
        R = np.linspace(0, Rmax, num = 1024) # input to afp function
        output = np.zeros(1024)
        
        for i, radius in enumerate(R):
            summedAfp = 0
            for j in neighbourDistances:
                summedAfp += (radius - (j / Rmin))
                # TODO add kaiser bessel smeared delta function to this
            # print summedAfp
            output[i] = summedAfp
        self.Afp = output
        
    def _afp(self, rMin, r, rNeighbour):
        print("yjdty")
        
    def generateAdjacencyMatrixNN(self, coords = None, knn = 10):    
        if coords is None:
            coords = self.cartesianCoords
            
        self.distanceMatrix = distance.pdist(coords, 'euclidean') # Create condensed distance matrix
        self.adjacencyMatrix = distance.squareform(self.distanceMatrix) # Create standard adjacency matrix
        
        for i, row in enumerate(self.adjacencyMatrix):
            lowestVals = np.partition(row, knn-1)[:knn] # take k neighbours from that row
            threshold = lowestVals.max()    # Take the longest distance from k neighbours
            exceedsThresholdFlags = row > threshold
            self.adjacencyMatrix[i][exceedsThresholdFlags] = 0
        return self.adjacencyMatrix
            
    def createMST(self):
        graph = nx.to_networkx_graph(self.adjacencyMatrix) # initialise graph
        self.mst = nx.minimum_spanning_tree(graph)
            
    def plot(self, coords):
        xs = coords[:,0]
        ys = coords[:,1]
        zs = coords[:,2]
        
        fig = plt.figure()
        ax = Axes3D(fig)
        
        ax.scatter(xs, ys, zs)

        ax.view_init(elev=50, azim=230)
        plt.show()

    def loadXyzFile(filePath):
        # Open a .xyz file from a file path
        with open (filePath, "r") as myfile: 
            data=myfile.readlines()
        
        data = data[2:] # Remove first two lines
        
        # split into terms and remove first term (the element name)
        for i, line in enumerate(data):
            data[i] = line.split()[1:]
        
        xs = []
        ys = []
        zs = []
        
        for line in data:
            xs.append(float(line[0]))
            ys.append(float(line[1]))
            zs.append(float(line[2]))

directory = '/home/cameron/Dropbox/T2_Dataset/molGeom/'
testFile = ['T2_1_num_molGeom.cif']
filePath = '/home/cameron/Dropbox/T2_Dataset/molGeom/T2_1_num_molGeom.cif'
outputFile = 'output.txt'
k = 15
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


with open(outputFile, 'w') as output:
    output.write('Cif file nearest distances\n')
    
for filename in os.listdir(directory):
    print(directory + filename)
    pc = PointCloud(directory + filename)
    
    pc.generateExtendedCloud(pc.lattice, reducedCloud=True)
    
    with open(outputFile, 'a') as output:
        output.write(str(filename) + " ")
        for i in range(k):
            output.write(str(np.around(pc.simplifiedMinimumDist[i], 7)) + " ")
        output.write("\n")
    pc.generateNetInput(pc.cartesianCoords, writeToFile =False)
        
    #pc.atomicFingerPrint(x[0])
    
    #fingerPrint = pc.atomicFingerPrint
    pc.plot(pc.cartesianCoords)
    pc.plot(pc.extendedCloud)
    

