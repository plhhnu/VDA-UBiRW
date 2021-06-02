#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
import random
from sklearn.metrics import auc, roc_curve, precision_recall_curve, f1_score, confusion_matrix


class DM_method:

    def __init__(self, interaction, positive_sample = None, Dsim = None):
        self.A = interaction
        if positive_sample:
            self.posivite_addr = positive_sample
        else:
            self.positive_addr = np.array(np.nonzero(interaction)).T
        self.negative_addr = np.array([(i, j) for i in range(interaction.shape[0]) for j in range(interaction.shape[1]) if interaction[i, j] == 0])
        self.positive_sample_size = int(self.positive_addr.size/2)
        self.negative_sample_size = int(self.negative_addr.size/2)
        self.SSD = Dsim
        self.label = []
        self.score = []
        self.K = 0
        self.valid_sample = None

    @property
    def is_get_result(self):
        if self.label and self.score:
            return True
        return False

    @property
    def auc(self):
        return self.performance(roc_curve)

    @property
    def aupr(self):
        return self.performance(precision_recall_curve, True)

    @property
    def get_label(self):
        if self.is_get_result:
            return self.label
        return None

    @property
    def get_score(self):
        if self.is_get_result:
            return self.score
        return None

    def performance(self, fun, ex=False):
        if not self.is_get_result:
            return None
        auct = 0
        for i in range(self.K):
            fpr, tpr, _ = fun(self.label[i], self.score[i])
            if ex:
                auct += auc(tpr, fpr)
            else:
                auct += auc(fpr, tpr)
        return auct/self.K


    def Kfoldcrossclassify(self, sample, K, fun="cv3"):
        r = []
        if fun != "cv3":
            m = np.mat(sample)
            if fun == "cv2":
                t = 0
            else:
                t = 1
            mt = self.Kfoldcrossclassify(np.array(range(np.max(m[:, t]) + 1)), K)
            # for i in range(K):
            r = [[j for j in sample if j[t] in mt[i]] for i in range(K)]
            return r

        l = sample.shape[0]
        t = sample.copy()
        n = math.floor(l / K)
        retain = l - n*K
        for i in range(K - 1):
            nt = n
            e = len(t)
            # if e % n and e % K:
            if retain > i:
                nt += 1
            a = random.sample(range(e), nt)
            r.append([t[i] for i in a])
            t = [t[i] for i in range(e) if (i not in a)]
        r.append(t)
        return r

    def prepare(self, K=0, cvt="cv3"):
        if K:
            self.K = K
            self.valid_sample = self.Kfoldcrossclassify(self.positive_addr, K, cvt)
        else:
            self.K = self.positive_addr
            self.valid_sample = np.array([np.array([i]) for i in self.positive_addr])

    def fun(self, A):
        return A

    def tarin(self, K=0, cvt="cv3"):
        self.label = []
        self.score = []
        self.prepare(K, cvt)
        for i in range(self.K):
            test = np.array(self.valid_sample[i])
            A = self.A.copy()
            A[test[:, 0], test[:, 1]] = 0
            self.label.append(np.array([1] * test.shape[0] + [0] * self.negative_sample_size))
            A = self.fun(A)
            sco_addr = np.vstack((test, self.negative_addr))
            self.score.append(A[sco_addr[:, 0], sco_addr[:, 1]])
    
    def train(self, K=0, cvt="cv3"):
        self.label = []
        self.score = []
        self.prepare(K, cvt)
        for i in range(self.K):
            test = np.array(self.valid_sample[i])
            neg_test = self.negative_addr[random.sample(range(self.negative_sample_size), test.shape[0])]
            A = self.A.copy()
            A[test[:, 0], test[:, 1]] = 0
            self.label.append(np.array([1] * test.shape[0] + [0] * neg_test.shape[0]))
            A = self.fun(A)
            sco_addr = np.vstack((test, neg_test))
            self.score.append(A[sco_addr[:, 0], sco_addr[:, 1]])

    def TP_FP_TN_FN(self, label, pre):
        l = label.size
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(l):
            if pre[i]:
                if label[i]:
                    TP += 1
                else:
                    FP += 1
            else:
                if label[i]:
                    FN += 1
                else:
                    TN += 1
        return TP, FN, FP, TN

    def acc_threshold(self,label, score):
        _, _, thresholds = roc_curve(label, score)
        acc_t = 0
        precision_arg = 0
        recall_arg = 0
        specificity_arg = 0
        f1 = 0
        l = len(thresholds)
        # print("{}".format(l))

        for i in thresholds:
            pre = score >= i
            a = confusion_matrix(label, pre, labels=[0, 1])
            TN, FP, FN, TP = a.ravel()
            f1 += f1_score(label, pre)
            if (TP + FP) != 0:
                pre_t = TP / (TP + FP)
                precision_arg += pre_t
            recall_arg += TP / (TP + FN)
            specificity_arg += TN / (TN + FP)
            acc_t += (TP+TN)/(TP+TN+FP+FN)
        return acc_t/l, precision_arg/l, recall_arg/l, specificity_arg/l, f1/l

    @property
    def evoluate(self):
        if not self.is_get_result:
            return None
        ev = np.zeros(5)
        for i in range(self.K):
            ev += np.array(self.acc_threshold(self.label[i], self.score[i]))
        return ev/self.K

    @property 
    def metrics_list(self):
        return list((self.auc, self.aupr)) + list(self.evoluate)
    
    @property
    def metrics_name(self):
        return ["AUC", "AUPR", "ACC", "Precision", "Recall", "Specificity", "F1 score"]

    @property
    def metrics_dict(self):
        return dict(zip(self.metrics_name, self.metrics_list))

    def predict(self):
        pre = self.fun(self.A)
        t = pre[self.negative_addr[:, 0], self.negative_addr[:, 1]]
        r = [(t[i], self.negative_addr[i][0], self.negative_addr[i][1]) for i in range(self.negative_sample_size)]
        r.sort(reverse=True)
        return r





if __name__ == '__main__':
    A = DM_method(np.eye(100))
    A.prepare()
    A.tarin(5)
    print(A.auc)
    print(A.label)
    print(A.score)
    print(A.evoluate)
    print(A.predict())
    print(A.metrics_dict)
