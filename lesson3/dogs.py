import numpy as np 
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

# avg heights plus standard normal distribution
# returns 2 arrays of numbers
grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

# creates histogram with greyhounds in red and labs in blue
plt.hist([grey_height,lab_height], stacked=True, color=['r','b'])
plt.show()