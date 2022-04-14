import os

import cv2
import h5py
import numpy as np
import skimage.color


def generateTrainingData(path, angRes: int, factor: int, label_size: int = None, is_test: bool = False):
    assert angRes in [1, 3, 5, 7, 9]

    label_size_h = label_size_w = label_size
    downRatio = 1 / factor
    sourceDatasets = os.listdir(path)
    if '.DS_Store' in sourceDatasets:
        sourceDatasets.remove('.DS_Store')
    # print(sourceDatasets)
    datasetsNum = len(sourceDatasets)
    idx = 0

    if not is_test:
        # SavePath = './TrainingData_' + str(angRes) + 'x' + str(angRes) + '_' + str(factor) + 'xSR/'
        SavePath = './data_train/'
    else:
        # SavePath = './TestingData_' + str(angRes) + 'x' + str(angRes) + '_' + str(factor) + 'xSR/'
        SavePath = './data_test/'
    if not os.path.exists(SavePath):
        os.makedirs(SavePath)

    for DatasetIndex in range(datasetsNum):
        sourceDataFolder = path + '/' + sourceDatasets[DatasetIndex] + '/training/'
        folders = os.listdir(sourceDataFolder)
        if '.DS_Store' in folders:
            folders.remove('.DS_Store')
        # print(folders)
        sceneNum = len(folders)
        # print(sceneNum)
        for iScene in range(sceneNum):
            idx_s = 0
            sceneName = folders[iScene]
            # print(sceneName)
            if not is_test:
                print('Generating training data of Scene_%s in Dataset %s......\t\t' % (sceneName[:-4], sourceDatasets[
                    DatasetIndex]))
            else:
                print('Generating testing data of Scene_%s in Dataset %s......\t\t' % (sceneName[:-4], sourceDatasets[
                    DatasetIndex]))
            dataPath = sourceDataFolder + '/' + sceneName
            data = h5py.File(dataPath, 'r')['LF'][:]

            data = np.transpose(data)

            U = data.shape[0]
            V = data.shape[1]
            LF = data[int(0.5 * (U - angRes + 2) - 1):int(0.5 * (U + angRes)),
                 int(0.5 * (V - angRes + 2) - 1): int(0.5 * (V + angRes)), :, :, 0: 3]
            U = LF.shape[0]
            V = LF.shape[1]
            H = LF.shape[2]
            W = LF.shape[3]

            if is_test:
                H = H // factor * factor
                W = W // factor * factor
                label_size_h = H
                label_size_w = W

            overlap_h = int((label_size_h - (H % label_size_h)) / (H // label_size_h))
            overlap_w = int((label_size_w - (W % label_size_w)) / (W // label_size_w))
            # print(overlap_h, overlap_w)
            start_h = [x * (label_size_h - overlap_h) for x in range(H // label_size_h)]
            start_h.append(H - label_size_h)
            start_h = set(start_h)
            start_w = [x * (label_size_w - overlap_w) for x in range(W // label_size_w)]
            start_w.append(W - label_size_w)
            start_w = set(start_w)

            for h in start_h:
                for w in start_w:
                    idx += 1
                    idx_s += 1
                    label = np.zeros((U, V, label_size_h, label_size_w))
                    data = np.zeros((U, V, int(label_size_h * downRatio), int(label_size_w * downRatio)))

                    for u in range(U):
                        for v in range(V):
                            tempHR = LF[u, v, h:(h + label_size_h), w: (w + label_size_w), :]
                            tempHRY = skimage.color.rgb2ycbcr(tempHR) / 255
                            tempHRY = tempHRY[:, :, 0]
                            label[u, v, :, :] = tempHRY
                            tempLRy = cv2.resize(tempHRY,
                                                 (int(label_size_w * downRatio), int(label_size_h * downRatio)),
                                                 interpolation=cv2.INTER_LINEAR)
                            data[u, v, :, :] = tempLRy
                    if not is_test:
                        SavePath_H5 = SavePath + '%06d' % idx + '.h5'
                    else:
                        SavePath_H5 = SavePath + sceneName[: -4] + '.h5'
                    label = np.transpose(label)
                    data = np.transpose(data).astype(np.float32)
                    f = h5py.File(SavePath_H5, 'w')
                    f.create_dataset('/data', data=data)
                    f.create_dataset('/label', data=label)
                    f.close()
            if not is_test:
                print(str(idx_s) + ' training samples have been generated\n')
            else:
                assert idx_s == 1
                print(str(idx_s) + ' testing samples have been generated\n')


if __name__ == '__main__':
    generateTrainingData(path='./data_mat', angRes=5, factor=4, label_size=320)
