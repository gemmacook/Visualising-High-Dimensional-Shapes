import numpy as np
import matplotlib.pyplot as plt 
import diffpy.Structure 
from scipy.spatial import distance
import networkx as nx

class PointCloud(object):
    def __init__(self, cifFilePath, superCell = False):
        self.filePath = cifFilePath
        self.molecule = diffpy.Structure.loadStructure(self.filePath)
        self.cartesianCoords = np.array(self.molecule.xyz_cartn)
        
    def loadCifFile(self, cifFilePath):
        self.filePath = cifFilePath
        self.molecule = diffpy.Structure.loadStructure(self.filePath)
        self.cartesianCoords = np.array(self.molecule.xyz_cartn)
        
    def createAtomicFingerPrint(self):
        print("adsfn")
        
    def createAdjacencyMatrixNN(self, coords = None, knn = 10):    
        if coords is None:
            coords = self.cartesianCoords
            
        self.distanceMatrix = distance.pdist(coords, 'euclidean') # Create condensed distance matrix
        self.adjacencyMatrix = distance.squareform(self.distanceMatrix) # Create standard adjacency matrix
        
        for i, row in enumerate(self.adjacencyMatrix):
            lowestVals = np.partition(row, knn-1)[:knn] # take k neighbours
            threshold = lowestVals.max()    # Take the longest distance from k neighbours
            exceedsThresholdFlags = row > threshold
            self.adjacencyMatrix[i][exceedsThresholdFlags] = 0
            
    def createMST(self):
        graph = nx.to_networkx_graph(self.adjacencyMatrix) # initialise graph
        self.mst = nx.minimum_spanning_tree(graph)
            
    def plot(self):
        xs = self.cartesianCoords[:,0]
        ys = self.cartesianCoords[:,1]
        zs = self.cartesianCoords[:,2]
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(xs, ys, zs)
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
