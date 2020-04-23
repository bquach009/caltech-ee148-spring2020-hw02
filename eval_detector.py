import os
import json
import numpy as np
import matplotlib.pyplot as plt

def rect_area(r_tl, c_tl, r_br, c_br):
    return max(r_br - r_tl + 1, 0) * max(c_br - c_tl + 1, 0)

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''

    # compute intersection area
    r_tl = max(box_1[0], box_2[0])
    c_tl = max(box_1[1], box_2[1])

    r_br = min(box_1[2], box_2[2])
    c_br = min(box_1[3], box_2[3])

    # Add 1 to each subtraction to include endpoints
    inter_area = rect_area(r_tl, c_tl, r_br, c_br)

    total_area = rect_area(box_1[0], box_1[1], box_1[2], box_1[3]) +\
    rect_area(box_2[0], box_2[1], box_2[2], box_2[3]) - inter_area
    iou = inter_area / total_area

    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.)
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives.
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        matched = [(pred[i][4] < conf_thr) for i in range(len(pred))]
        for i in range(len(gt)):
            found = False
            for j in range(len(pred)):
                if pred[j][4] >= conf_thr:
                    iou = compute_iou(pred[j][:4], gt[i])
                    if iou >= iou_thr and matched[j] == 0:
                        matched[j] = 1
                        TP += 1
                        found = True
                        break
            if not found:
                FN += 1
        for item in matched:
            if item == 0:
                FP += 1

    '''
    END YOUR CODE
    '''
    if conf_thr == 1:
        print(TP, FP, FN)

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = 'data/hw02_preds'
gts_path = 'data/hw02_annotations'

# load splits:
split_path = 'data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data.
'''
with open(os.path.join(preds_path,'weak_preds_train.json'),'r') as f:
    preds_train = json.load(f)

with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:

    '''
    Load test data.
    '''

    with open(os.path.join(preds_path,'weak_preds_test.json'),'r') as f:
        preds_test = json.load(f)

    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold.

confidence_thrs = []
for fname in preds_train:
    for item in preds_train[fname]:
        confidence_thrs.append(item[4])

confidence_thrs = np.sort(confidence_thrs) # using (ascending) list of confidence scores as thresholds
for threshold in [0.25, 0.5, 0.75]:
    print("Calculating for threshold:", threshold)
    tp_train = np.zeros(len(confidence_thrs))
    fp_train = np.zeros(len(confidence_thrs))
    fn_train = np.zeros(len(confidence_thrs))
    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=threshold, conf_thr=conf_thr)

    precision = np.divide(tp_train, tp_train + fp_train)
    recall = np.divide(tp_train, tp_train + fn_train)
    # Plot training set PR curves
    plt.step(recall.copy(), precision.copy())

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(["IOU = 0.25", "IOU = 0.5", "IOU = 0.75"])
plt.savefig("data/hw02_preds/weak_train_PR.jpg")

if done_tweaking:
    print('Code for plotting test set PR curves.')
    plt.figure()
    confidence_thrs = []
    for fname in preds_test:
        for item in preds_test[fname]:
            confidence_thrs.append(item[4])

    confidence_thrs = np.sort(confidence_thrs) # using (ascending) list of confidence scores as thresholds
    for threshold in [0.25, 0.5, 0.75]:
        print("Calculating for threshold:", threshold)
        tp_test = np.zeros(len(confidence_thrs))
        fp_test = np.zeros(len(confidence_thrs))
        fn_test = np.zeros(len(confidence_thrs))
        for i, conf_thr in enumerate(confidence_thrs):
            tp_test[i], fp_test[i], fn_test[i] = compute_counts(preds_test, gts_test, iou_thr=threshold, conf_thr=conf_thr)

        precision = np.divide(tp_test, tp_test + fp_test)
        recall = np.divide(tp_test, tp_test + fn_test)
        print(confidence_thrs[-5:])
        print(tp_test[-5:])
        print(fp_test[-5:])
        print(fn_test[-5:])
        print(precision[-5:])
        print(recall[-5:])
        # Plot training set PR curves
        plt.step(recall, precision)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(["IOU = 0.25", "IOU = 0.5", "IOU = 0.75"])
    plt.savefig("data/hw02_preds/weak_test_PR.jpg")
