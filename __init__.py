import math

import pandas as pd
import numpy as np
import random
import mysql.connector
from mysql.connector import Error
from mnist import MNIST
import os
import gzip
import matplotlib.pyplot as plt
import idx2numpy
import cv2 as cv
from NeuralNetworkOne.MySQL import MySQL


class NeuralNetwork(object):

    def __init__(self, rows, cols, server):
        self.server = server
        self.rows = rows
        self.cols = cols
        self.neurons = 10
        self.listsigmoid = []

    def initialize(self):
        for k in range(0, 10):
            arr = [[round(random.random() * random.choice((-1, 1)), 2) for i in range(self.cols)] for j in
                   range(self.rows)]
            i = 0
            j = 0
            while i < len(arr):
                j = 0
                while j < len(arr):
                    temp = self.server.insert_data(i, j, arr[i][j], k)
                    self.server.execute_query(temp)
                    print(arr[i][j])
                    j += 1
                i += 1
        print(pd.DataFrame(arr))

    def readPicture(self, i):
        picturefile = "train-images-idx3-ubyte"
        picturearr = idx2numpy.convert_from_file(picturefile)

        b = (picturearr[i] - np.min(picturearr[i])) / np.ptp(picturearr[i])
        return b

    def readLabel(self, i):
        labelfile = "train-labels.idx1-ubyte"
        labelarr = idx2numpy.convert_from_file(labelfile)
        return labelarr[i]

    def machineLearning(self, lastnum, curnum):
        while lastnum < curnum:
            self.listsigmoid = []
            temp = 0
            curpic = self.readPicture(lastnum)
            curlabel = self.readLabel(lastnum)
            while temp < self.neurons:
                self.calculateWeightedSum(curpic, temp)
                temp += 1
            print(self.calculateAccuracy(curlabel, self.listsigmoid))
            self.propogateone(curlabel, curpic, self.listsigmoid)
            print(max(self.listsigmoid))
            print((curlabel, self.listsigmoid.index(max(self.listsigmoid))))
            lastnum += 1

    def calculateWeightedSum(self, curpic, neuron):
        rows = 0
        cols = 0
        cursum = 0
        while rows < 28:

            cols = 0
            while cols < 28:
                curpicnumber = curpic[rows][cols].item()
                if (curpicnumber != 0):
                    curweight = self.server.getData(rows, cols, neuron)
                    cursum = cursum + curpicnumber * curweight
                cols += 1
            rows += 1
        self.listsigmoid.append(self.sigmoidAndBias(cursum, neuron))
        return cursum

    def sigmoidAndBias(self, cursum, neuron):
        bias = self.server.getBias(neuron)
        sigmoid = cursum - bias
        sigmoid = 1 / (1 + math.exp(-sigmoid))
        return sigmoid

    def calculateAccuracy(self, label, listsigmoid):
        curele = 0
        total = 0
        while curele < len(listsigmoid):
            if label == curele:
                total += (1 - listsigmoid[label]) ** 2
            else:
                total += listsigmoid[curele] ** 2
            curele += 1
        return total

    def propogateone(self, label, curpic, listsigmoid):
        curele = 0
        while curele < len(listsigmoid):
            if label == curele:

                templist = self.siftPic(curpic)
                search = 0
                while search < len(templist) / 2:
                    rows = templist[search][1]
                    cols = templist[search][2]
                    curweight = self.server.getData(rows, cols, curele)
                    bias = self.server.getBias(curele)
                    if (curweight < 0):
                        curweight * -1
                    else:
                        if listsigmoid[label] > 0.98:
                            pass
                        else:
                            if listsigmoid[label] < 0.5:
                                curweight = curweight * 1.55
                                bias -= 0.04
                            else:
                                curweight = curweight * 1.25
                                bias -= 0.015
                    curweight = round(curweight, 2)
                    self.server.updateBias(curele, bias)
                    self.server.updateElement(rows, cols, curele, curweight)
                    search += 1
            else:
                templist = self.siftPic(curpic)
                search = 0
                while search < len(templist) / 2:
                    rows = templist[search][1]
                    cols = templist[search][2]
                    curweight = self.server.getData(rows, cols, curele)
                    bias = self.server.getBias(curele)
                    if listsigmoid[curele] > 0.5:  # decrease by A LOT
                        curweight += curweight * -0.65
                        bias -= 0.057
                    else:  # decrease by a little
                        curweight += curweight * -0.85
                        bias += 0.010
                    curweight = round(curweight, 2)
                    self.server.updateBias(curele, bias)
                    self.server.updateElement(rows, cols, curele, curweight)
                    search += 1
            curele += 1

    def siftPic(self, curpic):
        rows = 0
        cols = 0
        listmax = []
        while rows < self.rows:
            cols = 0
            while cols < self.cols:
                if (curpic[rows][cols] != 0):
                    listmax.append([curpic[rows][cols], rows, cols])
                cols += 1
            rows += 1
        returnlist = sorted(listmax, key=lambda x: x[0], reverse=True)
        return returnlist

    def propogatetwo(self):
        pass


def main():
    server = MySQL("localhost", "root", "fake", "weights")

    test = NeuralNetwork(28, 28, server)
    #server.execute_query(server.databasevariables["addweightstable"])
    #test.initialize()
    test.machineLearning(100, 1000)


    # server.insert_bias(0, random.randint(0,10))
    # f = gzip.open('train-labels-idx1-ubyte.gz', 'r')
    # f.read(8)
    """for i in range(0, 10):
        buf = f.read(1)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        print(labels) """
    """f = gzip.open('train-images-idx3-ubyte.gz', 'r')
    image_size = 28
    num_images = 5
    f.read(16)
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)
    image = np.asarray(data[2]).squeeze()
    plt.imshow(image)
    plt.show() """

    # test.initialize()


if __name__ == "__main__":
    main()
