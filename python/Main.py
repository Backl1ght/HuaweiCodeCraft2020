import math
import datetime
import sys
import numpy as np

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Executor
# import time
import mmap

class Model:
    def __init__(
        self, 
        max_iters=0, 
        learning_rate=0, 
        X=[], 
        Y=[], 
        theta=[],
        optimization="GD"
    ):
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        self.X = X
        self.Y = Y
        self.feature_num = 0
        self.theta = theta
        self.optimization = optimization


    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))

    def normalization(self, X):
        for i in range(self.feature_num):
            X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:,i])
        return X

    def initParams(self):
        self.feature_num = self.X.shape[1]
        self.X = self.normalization(self.X)
        self.X = np.hstack((np.ones([self.X.shape[0], 1]), self.X))
        self.feature_num = self.X.shape[1]
        self.theta = np.zeros((self.feature_num), dtype=np.float)

    def compute_loss(self, y_pred, y):
        loss = - np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss

    def compute_gradient(self, X, y, theta):
        m = y.shape[0]
        y_pred = self.predict(X, theta)
        gradient = np.dot(y_pred - y, X) / m
        return gradient

    def predict(self, X, theta):
        y_pred = self.sigmoid(np.dot(X, theta))
        return y_pred

    def gradient_descent(self, X, y, theta, learning_rate):
        gradient = self.compute_gradient(X, y, theta)
        theta -= learning_rate * gradient
        return theta

    def adamGD(self, X, y, theta, v, u, t,
        learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        grad = self.compute_gradient(X, y, theta)
        grads = grad ** 2
        u = beta2 * u + (1 - beta2) * grads
        v = beta1 * v + (1 - beta1) * grad

        u_corrected = u / (1 - (beta2 ** t))
        v_corrected = v / (1 - (beta1 ** t))

        theta -= learning_rate / ((u_corrected ** 0.5) + epsilon) * v_corrected
        return theta, v, u

    def accuracy(self, y_pred, y):
        acc = np.mean(y_pred==y)
        return acc

    def train(self):
        self.initParams()
        if self.optimization == "GD":
            pass
        elif self.optimization == "Adam":
            v = np.zeros(self.feature_num)
            u = np.zeros(self.feature_num)
            t = 0

        for i in range(self.max_iters):
            # original
            if self.optimization == "GD":
                self.theta = self.gradient_descent(self.X, self.Y, self.theta, self.learning_rate)

            # adam gradient descent
            elif self.optimization =="Adam":
                t = t+1
                self.theta, v, u = self.adamGD(self.X, self.Y, self.theta, v, u, t, self.learning_rate)


def str2float(str):
    allInfo = str.strip().split(',')
    result = np.array(allInfo).astype(np.float)
    return result

def readFileLines1(filePath, flag, cnt):
    lines = []
    f = open(filePath, "rb")
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    while True:
        cnt = cnt - 1
        line = m.readline().decode('utf-8')
        if len(line) == 0 or (flag and cnt==0):
            break
        lines.append(line)
    m.close()
    return lines

def readFileLines2(filePath):
    lines = []
    f = open(filePath, "rb")
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    while True:
        line = m.readline().decode('utf-8')
        if len(line) == 0:
            break
        lines.append(line)
    m.close()
    return lines

def loadTrainSet(file_name, flag, cnt):
    lines = readFileLines1(file_name, flag, cnt)
    pool = ProcessPoolExecutor(max_workers=4)
    feats = list(pool.map(str2float, lines))
    feats = np.array(feats)
    return feats

def loadTestSet(file_name):
    lines = readFileLines2(file_name)
    pool = ProcessPoolExecutor(max_workers=4)
    feats = list(pool.map(str2float, lines))
    feats = np.array(feats)
    return feats

def loadResult(file_name):
    r = []
    lines = readFileLines2(file_name)
    for line in lines:
        r.append(int(float(line.strip())))
    r=np.array(r)
    return r


def savePredictResult(file_name, Y_test, separater):
    f = open(file_name, 'w')
    for i in range(len(Y_test)):
        f.write(str(Y_test[i])+separater)
    f.close()

if __name__ == "__main__":
    Local = False

    # start = time.time()
    if Local:
        print("Local mode")
        train_file =  "./data/train_data.txt"
        test_file = "./data/test_data.txt"
        predict_file = "./projects/student/result.txt"
        answer_file ="./projects/student/answer.txt"
    else:
        train_file =  "/data/train_data.txt"
        test_file = "/data/test_data.txt"
        predict_file = "/projects/student/result.txt"

    X = loadTrainSet(train_file, True, 2000)
    Y = X[:, -1]
    X = X[:,:-1]
    X_test = loadTestSet(test_file)
    X_test = np.hstack((np.ones([X_test.shape[0], 1]), X_test))

    # end1 = time.time()

    answer = np.array([])
    if Local:
        answer = loadResult(answer_file)

    model = Model(
        max_iters=1, 
        learning_rate=5,
        X=X,
        Y=Y,
        optimization="GD"
    )
    model.train()

    Y_test = model.predict(X_test, model.theta)
    Y_test = np.round(Y_test).astype(int)
    savePredictResult(predict_file, Y_test, "\n")

    # end2 = time.time()

    # if Local:
    #     a = loadResult(answer_file)
    #     p = loadResult(predict_file)

    #     print("answer lines:%d" % (len(a)))
    #     print("predict lines:%d" % (len(p)))

    #     errline = 0
    #     for i in range(len(a)):
    #         if a[i] != p[i]:
    #             errline += 1

    #     accuracy = (len(a)-errline)/len(a)
    #     print("accuracy:%f" %(accuracy))
    #     print("read time:%.2fs" % (end1 - start))
    #     print("train time:%.2fs" % (end2 - end1))