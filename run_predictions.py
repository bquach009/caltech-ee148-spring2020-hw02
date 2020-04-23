import os
import numpy as np
import json
from PIL import Image

def compute_convolution(I, T, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays)
    and returns a heatmap where each grid represents the output produced by
    convolution at each location. You can add optional parameters (e.g. stride,
    window_size, padding) to create additional functionality.
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)

    '''
    BEGIN YOUR CODE
    '''
    # Extract height and width of the template
    h, w, _ = T.shape

    # adjusted height of box
    if h == 62:
        adj_h = 20
    else:
        adj_h = round(h / 2)

    # Create template vector to convolve
    temp = T.flatten()
    norm = np.linalg.norm(temp)
    if norm > 0:
        temp = temp / norm

    # Use three channels to store
    # - heatmap score (i.e. confidence score)
    # - height of template
    # - width of template
    heatmap = np.zeros((n_rows, n_cols, 3))
    for i in range(n_rows - h):
        for j in range(n_cols - w):
            imag = I[i:i+h, j:j+w, :].flatten()
            norm = np.linalg.norm(imag)
            if norm > 0:
                imag = imag / norm
            heatmap[i, j, :] = [np.dot(temp, imag), adj_h, w]

    '''
    END YOUR CODE
    '''


    return heatmap


def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''

    '''
    Pseudocode:

    for each pixel in heatmap:
        if confidence score above T:
            if new box doesn't intersect with any previous boxes:
                add new box to list using the stored width and heights
    '''

    # Define a threshold
    T = 0.92

    rows, cols, _ = heatmap.shape
    # Assume traffic lights in top half of the picture

    for i in range(rows // 2):
        for j in range(cols):
            score = heatmap[i, j, 0]
            if score >= T:
                intersects = False
                for item in output:
                    if i >= item[0] and i <= item[2] and \
                    j >= item[1] and j <= item[3]:
                        intersects = True
                        break
                if not intersects:
                    h = heatmap[i, j, 1]
                    w = heatmap[i, j, 2]
                    output.append([i, j, i + h, j + w, score])

    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>.
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>.
    The first four entries are four integers specifying a bounding box
    (the row and column index of the top left corner and the row and column
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1.

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    output = []
    # You may use multiple stages and combine the results
    # read image using PIL:
    temp = Image.open("data/RedLights2011_Medium/RL-001.jpg")

    # convert to numpy array:
    temp = temp.crop((316, 154, 323, 171))

    # Extract different size image
    temp2 = Image.open("data/RedLights2011_Medium/RL-010.jpg")
    temp2 = temp2.crop((325, 30, 345, 92))

    for filter in [temp, temp2]:
        T = np.asarray(filter).copy()

        heatmap = compute_convolution(I, T)
        result = predict_boxes(heatmap)
        for item in result:
            if item not in output:
                output.append(item)
    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = 'data/RedLights2011_Medium'

# load splits:
split_path = 'data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = 'data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''
preds_train = {}
n = len(file_names_train)
for i in range(len(file_names_train)):
    print("Predicting: {}/{}".format(i, n))

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'weak_preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set.
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'weak_preds_test.json'),'w') as f:
        json.dump(preds_test,f)
