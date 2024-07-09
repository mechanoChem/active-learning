import pickle
import matplotlib.pyplot as plt

# Load the figure using pickle
with open('3d_surface_plot.pkl', 'rb') as f:
    fig = pickle.load(f)

# Show the figure
plt.show()