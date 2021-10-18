##### This script will refine the predictions based on detected object by the object detector. Following by the work of https://github.com/vt-vl-lab/iCAN#######


import numpy as np
import pickle

with open('../infos/prior.pickle', 'rb') as file:  # Binary read
    u = pickle._Unpickler(file)
    u.encoding = 'latin1'
    priors = u.load()


def apply_prior(Object, prediction_HOI_in):
    # print('Object : ', Object)                          # class_ids_extended[1:]
    prediction_HOI = np.ones(prediction_HOI_in.shape)  # ones는 zeros와 마찬가지로 1로 가득찬 array를 생성
    # print('prediction_HOI : ',prediction_HOI)
    for index, prediction in enumerate(prediction_HOI):
        # print('Object[index] : ', Object[index])
        prediction_HOI[index] = priors[int(Object[index])]
    return prediction_HOI


if __name__ == '__main__':
    res = {}
    for k in range(80):
        prediction_HOI = np.ones((1, 29))
        res[k] = apply_prior([k], prediction_HOI)
