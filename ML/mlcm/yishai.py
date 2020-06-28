import mlcm

import numpy

y_test = [(0,1),(0,1)]
y_pred = [(0,1),(1,0)]

classes = ['prayer', 'procedures-for-judges-and-conduct-towards-them', 'learning', 'kings', 'hilchot-chol-hamoed', 'laws-of-judges-and-courts', 'laws-of-animal-sacrifices', 'financial-ramifications-of-marriage', 'idolatry', 'laws-of-transferring-between-domains']
# ['zero','one']

y_test = numpy.array(y_pred)
y_pred = numpy.array(y_test)

cm = mlcm.confusion_matrix(y_test, y_pred)
mlcm.plot_confusion_matrix(cm, classes, normalize=False)
# cm = mlcm.draw_cm(y_test, y_pred, classes, normalize=False)
