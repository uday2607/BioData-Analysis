import matplotlib.pyplot as plt

figure = plt.gcf()  # get current figure
figure.set_size_inches(32, 16)

n = 3
a = 2
b = 2

ax1 = plt.subplot2grid(shape=(a,b), loc=(0,0))
ax2 = plt.subplot2grid((a,b), (0,3))
ax3 = plt.subplot2grid((a,b), (0,6))
ax4 = plt.subplot2grid((a,b), (2,1))
ax5 = plt.subplot2grid((a,b), (2,4))

plt.show()
