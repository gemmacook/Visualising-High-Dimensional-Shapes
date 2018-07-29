import matplotlib.pyplot as plt
import numpy as np
import random
import copy

epsilon = .2
minpts = 4

#create data
Data = []
Names = []
Labels = []
data = open('ppc5688.txt', 'r')
for line in data:
   Names.append(line.replace('\n','').split(" ")[0])
   Data.append((line.split(" ")[1].replace(' ',''), line.split(" ")[2].replace(' ','')))
   Labels.append(0)
Data = random.sample(Data, 500)
Data = np.asarray(Data).astype(np.float)
#Data = StandardScaler().fit_transform(Data)

plt.figure(figsize=(7.5,7.5))
scatter_axis = plt.gca()
scatter_axis.set_xlim([np.min(Data), np.max(Data)])
scatter_axis.set_ylim([np.min(Data), np.max(Data)])
scatter_axis.plot(Data[:,0], Data[:,1], '.', color='black', markeredgecolor='black')
plt.xlabel('First Minimum Distance')
plt.ylabel('Second Minimum Distance')
plt.ion()
plt.show() 

def distance(p,q):
   d = np.linalg.norm(p-q)
   return d

def plotcluster(neighbours, colour):
   for n in neighbours:
      circle1 = plt.Circle((n[0], n[1]), radius=epsilon, color=colour, alpha=0.1)
      scatter_axis.plot(n[0], n[1], '.', color=colour, markeredgecolor='none')
      scatter_axis.add_artist(circle1)
      plt.pause(0.0001)

def extendcluster(Data, neighbours,colour):
   for n in neighbours:
      newneighbours = [x for x in Data if distance(x, n) < epsilon]
      plotcluster(newneighbours, colour)
      if n in Data:
         Data = np.delete(Data, np.where(Data == n)[0][0], 0)
         #print 'unclassified points ', Data.shape[0]
      Data = extendcluster(Data, newneighbours, colour)
      if newneighbours == []:
         return Data 
      else:
         newneighbours.pop() 
   return Data   

Run = True 
clustercount = 0


while Run:
   Run = False
   for seed in Data:
      colour = list(np.random.rand(3))
      neighbours = [x for x in Data if distance(x, seed) < epsilon]
      if len(neighbours) >= minpts:
         clustercount += 1
         print 'Cluster ', clustercount
         Data = extendcluster(Data, neighbours, colour)
         Run = True       
         #plt.savefig('gemmaDBSCAN'+str(clustercount)+'.pdf')

print Data.shape
plt.ioff()
plt.show()



