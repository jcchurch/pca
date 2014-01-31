import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

faceFilenames = [
     'face11.png'
    ,'face12.png'
    ,'face13.png'
    ,'face14.png'
    ,'face15.png'
    ,'face21.png'
    ,'face22.png'
    ,'face23.png'
    ,'face24.png'
    ,'face25.png'
    ,'face31.png'
    ,'face32.png'
    ,'face33.png'
    ,'face34.png'
    ,'face35.png'
    ,'face41.png'
    ,'face42.png'
    ,'face43.png'
    ,'face44.png'
    ,'face45.png'
]

def loadImages(filenames):
    images = []
    for filename in filenames:
        print filename
        a = mpimg.imread(filename)
        if len(a.shape) == 3:
            images.append(a[:,:,0])
        if len(a.shape) == 2:
            images.append(a[:,:])
    return images

def averageImage(images):
    return sum(images)/len(images)

def showImage(image):
    imgplot = plt.imshow(image);
    imgplot.set_cmap('Greys');
    plt.show()

def buildA(images, avgimg):
    n = avgimg.shape[0]*avgimg.shape[1]
    A = np.zeros((n, len(images)))
    i = 0
    for image in images:
        diffImage = image - avgimg
        diffImageCol = diffImage.reshape((n))
        A[:,i] = diffImageCol
        i += 1
    return A

def eigenFaces(A):
    (dimSquared, nimages) = A.shape
    dim = np.sqrt(dimSquared)
    M = np.dot(A.T, A)
    [w, v] = np.linalg.eig(M)
    eigFaces = []
    for i in range(nimages):
        face = np.dot(A, v[:,i]).reshape((dim,dim))
        eigFaces.append(face)
    return eigFaces

def buildU(A, top=5):
    (dimSquared, nimages) = A.shape
    M = np.dot(A.T, A)
    [w, v] = np.linalg.eig(M)
    indices = np.argsort(w)

    U = np.zeros((dimSquared, top))
    j = 0
    for i in range(nimages):
        if indices[i] >= (nimages - top):
            U[:,j] = np.dot(A, v[:,i])
            j += 1
    return U

def computeWeights(A, U):
    W = np.dot(U.T, A)
    return W

def findMatch(unknownImage, avgimg, A, U):
    n = avgimg.shape[0]*avgimg.shape[1]
    unknownAdj = unknownImage.reshape((n,1))
    avgimgAdj = avgimg.reshape((n,1))
    W = computeWeights(A, U)

    unknown_weights = np.dot(U.T, (unknownAdj - avgimgAdj))
    unknown_weights_tiled = np.tile(unknown_weights, (1, A.shape[1]))

    comparison = np.sum( (W - unknown_weights_tiled)**2, 0 )
    ranks = np.argsort(comparison)
    return ranks.indexOf(0)

def doWork():
    images = loadImages(train)
    avgimg = averageImage(images)
    A = buildA(images, avgimg)
    eigFaces = eigenFaces(A)
    U = buildU(A)
    return [images, avgimg, A, eigFaces, U]
