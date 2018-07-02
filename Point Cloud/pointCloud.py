import matplotlib.pyplot as plt 

with open ("AABHTZ.xyz", "r") as myfile: # Open a .xyz file from the same directory 
    data=myfile.readlines()

data = data[2:]

for i, line in enumerate(data):
    data[i] = line.split()[1:] # split into terms and remove first term
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xs = []
ys = []
zs = []

for line in data:
    xs.append(float(line[0]))
    ys.append(float(line[1]))
    zs.append(float(line[2]))

ax.scatter(xs, ys, zs)
plt.show()


