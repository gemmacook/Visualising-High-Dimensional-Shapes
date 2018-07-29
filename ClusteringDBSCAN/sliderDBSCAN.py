from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn import metrics

#create data
Data = []
Names = []

data = open('PPC_Equivalency_Test10.txt', 'r')
initial_epsilon, initial_minpoints, initial_dimension = 0.5, 3, 2

for line in data:
   Names.append(line.replace('\n','').split(" ")[0])
   Data.append((line.replace('\n','').split(" ")[1:11]))
Data = np.asarray(Data).astype(np.float)
#Data = StandardScaler().fit_transform(Data)

Data = np.asarray(Data).astype(np.float)

plt.figure(figsize=(7.5,7.5))
scatter_axis = plt.gca()
scatter_axis.set_xlim([np.min(Data), np.max(Data)])
scatter_axis.set_ylim([np.min(Data), np.max(Data)])

#create axis for gui selectors
axdims = plt.axes([0.25, 0.90, 0.5, 0.02], facecolor='white')
axeps = plt.axes([0.25, 0.025, 0.5, 0.02], facecolor='white')
axminpts = plt.axes([0.25, 0.055, 0.5, 0.02], facecolor='white')
resetax = plt.axes([0.85, 0.025, 0.05, 0.02])

dimension = Slider(axdims, 'Dimension', 2, 10.0, valinit=initial_epsilon)
epsilon = Slider(axeps, 'Eps', 0.1, 2.0, valinit=initial_epsilon)
minpoints = Slider(axminpts, 'minPts', 0.1, 30.0, valinit=initial_minpoints)
resetbutton = Button(resetax, 'Reset', color='white', hovercolor='0.975')

def rundbscan(epsilon, minpoints, dimension):
   data = Data[:,0:dimension+1]
   db = DBSCAN(eps=epsilon, min_samples=minpoints).fit(data)
   core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
   core_samples_mask[db.core_sample_indices_] = True
   labels = db.labels_
   
   # Number of clusters in labels, ignoring noise if present.
   n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
     
   # Black removed and is used for noise instead.
   unique_labels = set(labels)
   colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
   for k, col in zip(unique_labels, colors):
      marker = '.'
      if k == -1:
        # Black x used for noise.
        col = [0, 0, 0, 1]
        marker = '.'
      scatter_axis.set_xlim([0,np.amax(Data)])
      scatter_axis.set_ylim([0,np.amax(Data)])
      class_member_mask = (labels == k)

      #silhouette_avg = silhouette_score(Data, labels)
      silhouette_avg = 0
      pca = PCA(n_components=2, whiten=True)
      DataPCA = pca.fit_transform(data)
      xy = Data[class_member_mask & core_samples_mask]
      scatter_axis.plot(xy[:, 0], xy[:, 1], marker, markerfacecolor=tuple(col), markeredgecolor='none')

      xy = Data[class_member_mask & ~core_samples_mask]
      scatter_axis.plot(xy[:, 0], xy[:, 1], marker, markerfacecolor=tuple(col), markeredgecolor='none')

   print 'Dimension = ', dimension, ' Epsilon = ', round(epsilon,2), ' MinPts = ', round(minpoints,2), 'Estimated number of clusters = ', n_clusters_
   plt.suptitle('Estimated number of clusters: %d' % (n_clusters_), fontsize=12)
   plt.draw()
   plt.savefig('SliderResults/slider_eps_'+str(epsilon).replace('.','')+'minpts_'+str(minpoints).replace('.','')+'.pdf')

def update(val):
   scatter_axis.clear()
   rundbscan(epsilon.val, round(minpoints.val), int(round(dimension.val)))
  
def reset(event):
   epsilon.reset()
   minpoints.reset()
   dimension.reset()

#call update function when slider changes
epsilon.on_changed(update)
minpoints.on_changed(update)
dimension.on_changed(update)
#call reset function when reset button clicked
resetbutton.on_clicked(reset)

rundbscan(initial_epsilon, initial_minpoints, initial_dimension)
plt.show()
