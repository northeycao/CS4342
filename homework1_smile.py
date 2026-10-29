import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

y = np.array([1, 1, 1, 0, 0])
yhat = np.array([0, 1, 0, 1, 1])

def fPC (y, yhat):
    f = (y == yhat)
    return np.mean(f)

# print(fPC(y, yhat))


def measureAccuracyOfPredictors (predictors, X, y):
    num_p = len(predictors)
    num_i = len(X)
    predictions = np.zeros([num_p, num_i])
    # for image in X:
    #     for p in predictors:
    #         predictions[image, p] = X[image, p[0], p[1]] > X[image, p[2], p[3]]

    for p in range(num_p):
        predictions[p] = X[:, predictors[p][0], predictors[p][1]] > X[:, predictors[p][2], predictors[p][3]]

    yhat = np.mean(predictions, axis=0) > 0.5
    return fPC(y, yhat)


def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):
    show = False
    if show:
        # Show an arbitrary test image in grayscale
        im = testingFaces[0,:,:]
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        # Show r1,c1
        rect = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Show r2,c2
        rect = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        # Display the merged result
        plt.show()

    n = trainingFaces[0].shape[0]
    predictors = np.zeros((np.ceil(n*n / 2)))

    coordinates = np.zeros((n, 2))


    for i1 in range(n):
        for j1 in range(n): #pixel = i*n + j
            for i2 in range(n - i1):
            predictors[i*n+j] =
    for i in range (6):
        measureAccuracyOfPredictors(predictors, trainingFaces, trainingLabels)


def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")



predictors = [[1, 2, 1, 3], [1, 2, 1, 4], [1, 2, 1, 5]]
faces, labels = loadData("train")
print(measureAccuracyOfPredictors(predictors, faces, labels))