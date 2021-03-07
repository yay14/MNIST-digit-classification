# MNIST-digit-classification

MNIST ("Modified National Institute of Standards and Technology") is the classic dataset of handwritten images and it has served as the basis for benchmarking classification algorithms.It has a training set of 60,000 examples, and a test set of 10,000 examples.MNIST is a subset of the larger set available from NIST.

A Bayesian classifier is built to classify the numbers in the MNIST handwritten digit recognition databases.The main tasks are to:

i) distinguish between ‘0’ and ‘1’ digits
  Examples from MNIST dataset for '0' and '1' are
      ![](https://github.com/yay14/MNIST-digit-classification/blob/main/images/zero.png)
      ![](https://github.com/yay14/MNIST-digit-classification/blob/main/images/one.png)

ii) distinguish between ‘3’ and ‘8’ digits
 Examples from MNIST dataset for '3' and '8' are
      ![](https://github.com/yay14/MNIST-digit-classification/blob/main/images/three.png)
      ![](https://github.com/yay14/MNIST-digit-classification/blob/main/images/eight.png)

Next,the classification accuracy of the model is calculated and the ROC curves between GAR (Genuine Acceptance Rate) and FAR(False Acceptance Rate) curves are plotted for model evaluation purposes.

## To Run

To run the model locally, fist clone the repository using following command in your terminal.

git clone https://github.com/yay14/MNIST-digit-classification.git

The dataset is zipped inorder to compress it to uploadable size, so start by unzipping the compressed binary files or download the dataset from http://yann.lecun.com/exdb/mnist/
After downloading the zipped file, the binary files need to be extracted and uploaded in the same folder as the python notebook.
Then open the MNIST digits Classification.ipynb file and run in Jupyter notebook or google colab.

## Dataset

The MNIST dataset for Handwritten digit recognition is stored in IDX Format which is difficult to process so we need to convert it to Python Numpy Array.The following code snippet converts the data from binary format to numpy array using datatype conversion.

    #..........................................................For training dataset..............................................................
    print("Training Dataset.......")

    for name in trainingfilenames.keys():
     if name == 'images':
      train_imagesfile = open(trainingfilenames[name],'rb')
     if name == 'labels':
      train_labelsfile = open(trainingfilenames[name],'rb')#,encoding='latin-1')

    train_imagesfile.seek(0)
    magic = st.unpack('>4B',train_imagesfile.read(4))
    if(magic[0] and magic[1])or(magic[2] not in data_types):
     raise ValueError("File Format not correct")

    #Information
    nDim = magic[3]
    print("Data is "+str(nDim)+"-D")
    dataType = data_types[magic[2]][0]
    print("Data Type :: ",dataType)
    dataFormat = data_types[magic[2]][1]
    print("Data Format :: ",dataFormat)
    dataSize = data_types[magic[2]][2]
    print("Data Size :: "+str(dataSize)+" byte\n")


    #offset = 0004 for number of images
    #offset = 0008 for number of rows
    #offset = 0012 for number of columns
    #32-bit integer (32 bits = 4 bytes)
    train_imagesfile.seek(4)
    nImg = st.unpack('>I',train_imagesfile.read(4))[0] #num of images/labels
    nR = st.unpack('>I',train_imagesfile.read(4))[0] #num of rows
    nC = st.unpack('>I',train_imagesfile.read(4))[0] #num of columns

    train_labelsfile.seek(8) #Since no. of items = no. of images and is already read
    print("no. of images :: ",nImg)
    print("no. of rows :: ",nR)
    print("no. of columns :: ",nC)
    print
    #Training set
    #Reading the labels
    train_labels_array = np.asarray(st.unpack('>'+dataFormat*nImg,train_labelsfile.read(nImg*dataSize))).reshape((nImg,1))
    #Reading the Image data
    nBatch = 10000
    nIter = int(math.ceil(nImg/nBatch))
    nBytes = nBatch*nR*nC*dataSize
    nBytesTot = nImg*nR*nC*dataSize
    train_images_array = np.array([])
    for i in range(0,nIter):
     #try:
     temp_images_array = np.asarray(st.unpack('>'+dataFormat*nBytes,train_imagesfile.read(nBytes))).reshape((nBatch,nR,nC))
     '''except:
      nbytes = nBytesTot - (nIter-1)*nBytes
      temp_images_array = 255 - np.asarray(st.unpack('>'+'B'*nbytes,train_imagesfile.read(nbytes))).reshape((nBatch,nR,nC))'''
     #Stacking each nBatch block to form a larger block
     if train_images_array.size == 0:
      train_images_array = temp_images_array
     else:
      train_images_array = np.vstack((train_images_array,temp_images_array))
     temp_images_array = np.array([])

    print("Training Set Labels shape ::",train_labels_array.shape)
    print("Training Set Image shape ::",train_images_array.shape)

The same is done for testing dataset as well.

## Data pre-processing

As we need only a subset of the whole dataset, we will now filter out those images which are labelled as one of the following [0,1,3,8].
    
Model 1:

    #counting test samples which are either 0 or 1
    c=0
    for i in range(10000):
        if (test_labels_array[i]==1) or (test_labels_array[i]==0) :
            c=c+1
    print(c)
 
 Model 2:
 
    #counting test samples which are either 3 or 8
    c=0
    for i in range(10000):
        if (test_labels_array[i]==3) or (test_labels_array[i]==8) :
            c=c+1
    print(c)

Since the image is in 2d array format, we need to flatten it before model training.

    #preparing test dataset
    x_test =np.ndarray(shape=(c,784),dtype='int32')
    y_test = np.ndarray(shape=(c),dtype='int32')
    j=0
    for i in range(10000):
        if ((test_labels_array[i]==1) or (test_labels_array[i]==0)) and j<c:
            x_test[j]=test_images_array[i].flatten()
            y_test[j]=test_labels_array[i]
            j=j+1
        
## Classification


    #Classification using Gaussian Naive Bayes
    predicted = GNB_classifier.predict(x_test)
    print(predicted)
    
## Results

    #Summarising results
    print("\nClassification report for classifier %s:\n%s\n" % (GNB_classifier, metrics.classification_report(y_test, predicted)))
    disp = metrics.plot_confusion_matrix(GNB_classifier, x_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print("\nConfusion matrix:\n%s" % disp.confusion_matrix)
    print("\nAccuracy of the Algorithm: ", GNB_classifier.score(x_test, y_test))
    plt.show()

## Comparison

Both models are trained on same lines but on different data samples. Therefore, the results vary significantly.

**Accuracy**

Model 1: 98.77%

Model 2: 69.91%

**Confusion Matrix**

Model 1:
    ![](https://github.com/yay14/MNIST-digit-classification/blob/main/images/CM1.png)

Model 2:
    ![](https://github.com/yay14/MNIST-digit-classification/blob/main/images/CM2.png)

**ROC curve**

Model 1:
    ![](https://github.com/yay14/MNIST-digit-classification/blob/main/images/ROC1.png)

Model 2:
    ![](https://github.com/yay14/MNIST-digit-classification/blob/main/images/ROC2.png)


