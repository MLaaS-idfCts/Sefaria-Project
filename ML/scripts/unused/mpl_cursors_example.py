# Here is an example to illustrate how to use mplcursors for an sklearn confusion matrix.

# Unfortunately, mplcursors doesn't work with seaborn heatmaps. Seaborn uses a QuadMesh for the heatmap, which doesn't support the necessary coordinate picking.

# In the code below I added the confidence at the center of the cell, similar to seaborn's. I also changed the colors the texts and arrows to be easier to read. You'll need to adapt the colors and sizes to your situation.

from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import mplcursors

y_true = ["cat", "ant", "cat", "cat", "ant", "bird", "dog"]
y_pred = ["ant", "ant", "cat", "cat", "ant", "cat", "dog"]
labels = ["ant", "bird", "cat", "dog"]
confusion_mat = confusion_matrix(y_true, y_pred, labels=labels)

fig, ax = plt.subplots()
heatmap = plt.imshow(confusion_mat, cmap="jet", interpolation='nearest')

for x in range(len(labels)):
    for y in range(len(labels)):
        ax.annotate(str(confusion_mat[x][y]), xy=(y, x),
                    ha='center', va='center', fontsize=18, color='white')

plt.colorbar(heatmap)
plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)
plt.ylabel('Predicted Values')
plt.xlabel('Actual Values')

cursor = mplcursors.cursor(heatmap, hover=True)
@cursor.connect("add")
def on_add(sel):
    i, j = sel.target.index
    sel.annotation.set_text(f'{labels[i]} - {labels[j]} : {confusion_mat[i, j]}')
    sel.annotation.set_fontsize(12)
    sel.annotation.get_bbox_patch().set(fc="papayawhip", alpha=0.9, ec='white')
    sel.annotation.arrow_patch.set_color('white')

# plt.ioff()
# plt.gcf().show()
# plt.ioff()
# heatmap.show()
# plt.show()
plt.show(block=True)