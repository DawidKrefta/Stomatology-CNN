from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

binary1 = np.array([[21, 4],
                   [12, 12]])
class_names = ['live', 'die']

fig, ax = plot_confusion_matrix(conf_mat=binary1,
                                colorbar=True,
                                class_names=class_names)
plt.show()
