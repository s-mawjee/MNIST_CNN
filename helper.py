import numpy as np
import pymysql.cursors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itertools

classes = [0,1,2,3,4,5,6,7,8,9]

def get_data_from_database(test_data=False):
    """
    Get hand written digitals from the database
    """
    
    img_set = []
    label_set = []
    
    # Connect to the database
    connection = pymysql.connect(host='',
                                 user='',
                                 password='',
                                 db='mnist02',
                                 cursorclass=pymysql.cursors.DictCursor)
    
    sql = "SELECT data FROM mnist_train"
    
    if(test_data):
        sql = "SELECT data FROM mnist_test"
    
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            results = cursor.fetchall()
            for row in results :
                img = [int(x) for x in row['data'].split(',')]
                label = img[0]
                img = img[1:]
                reshaped_img = np.reshape(img,[28, 28])
                img_set.append(reshaped_img)
                label_set.append(label)
    finally:
        connection.close()
    
    return np.asarray(img_set, dtype=np.float32), np.asarray(label_set, dtype=np.float32)

def get_data_from_file(test_data=False):
    img_set = []
    label_set = []

    file = 'data/mnist_train.txt'
    if(test_data):
        file = 'data/mnist_test.txt'
    
    with open(file) as training_data:
        for line in training_data:
            data = line.split(';')[1].replace('\n','').replace('"','')
            img = data.split(',')
            label = img[0]
            img = img[1:]
            reshaped_img = np.reshape(img,[28, 28])
            img_set.append(reshaped_img)
            label_set.append(label)
    return np.asarray(img_set, dtype=np.float32), np.asarray(label_set, dtype=np.float32)

def show_set_of_images(data, label, count=4):
    for i in range(count):
        plt.subplot(count,1,i+1)
        plt.imshow(data[i],cmap='gray')

def plot_histogram(data, name):
    fig, ax = plt.subplots()
    plt.hist(data,bins=[0,1,2,3,4,5,6,7,8,9,10])
    ax.set_title(name)

def plot_confusion_matrix(cm, normalise=False):
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    if(normalise):
        plt.title('Confusion matrix (Normalised)')
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if(normalise):
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, decimals=2)
        print("Confusion matrix (Normalised)")
    else:
        print('Confusion matrix')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
