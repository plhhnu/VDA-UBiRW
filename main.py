from progress.bar import Bar
import numpy as np
from functools import partial
from multiprocessing import Pool
import re
import pandas as pd
from DM_BiRW import BiRW
from DM_UBiRW import UBiRW
import sys
sys.path.append("../Drug_Virus")
sys.path.append("./Drug_Virus")

pattern = re.compile("'(.*)'")

dset = None


def train_thread(func):
    func, cvt = func
    x = func()
    x.tarin(5, cvt)
    auct = x.auc
    aupr = x.aupr
    acc_t, precision_arg, recall_arg, specificity_arg, f1 = x.evoluate
    return auct, aupr, acc_t, precision_arg, recall_arg, specificity_arg, f1


def train(func, cv="cv3", cc=0) -> pd.Series:
    pool = Pool(processes=10)
    fun_str = pattern.findall(str(func))[0].split(".")[1]
    bar = Bar(fun_str + " training", max=100, suffix='%(percent)d%%')
    o = pd.DataFrame(
        columns=["ACC", "Precision", "Recall", "Specifity", "AUC", "AUPR", "f1 score"])
    update = lambda a: bar.next()
    result = []
    for i in range(100):
        result.append(pool.apply_async(
            train_thread, ((func, cv),), callback=update))

    bar.finish()
    pool.close()
    pool.join()
    for res in result:
        auct, aupr, acc_t, precision_arg, recall_arg, specificity_arg, f1 = res.get()
        o = o.append(pd.Series([acc_t, precision_arg, recall_arg, specificity_arg, auct, aupr, f1],
                               index=["ACC", "Precision", "Recall", "Specifity", "AUC", "AUPR", 'f1 score']), ignore_index=True)

    bar.finish()
    avg = o.mean()
    print(pd.DataFrame(avg).T.to_string(index=None))
    o = o.append(avg, ignore_index=True)
    if (cc == 0):
        o.to_excel("{}_{}_{}.xlsx".format(fun_str, dset, cv))
    else:
        o.to_excel("{}_{}_{}_{}.xlsx".format(fun_str, cc, dset, cv))
    return avg


if __name__ == "__main__":
    datasets = ["dataset_1", "dataset_2", "dataset_3"]
    for dataset in datasets:
        print(dataset)
        filename = dataset + "/associate.npy"
        d_filename = dataset + "/drug_sim.npy"
        v_filename = dataset + "/virus_sim.npy"

        a = np.load(filename, allow_pickle=True)
        d = np.load(d_filename, allow_pickle=True)
        v = np.load(v_filename, allow_pickle=True)

        dset = dataset

        if dataset == "dataset_1":
            UBiRW = partial(UBiRW, A=a, SD=d, SV=v, **{'alpha': 0.3, 'l1': 11, 'l2': 11,
                                'dlalpha': 0.1, 'vlalpha': 0.1, 'lam': 2.5, 'beta': 0.30000000000000004})
            BiRW = partial(BiRW, A=a, SD=d, SV=v, **{
                            'alpha': 0.6, 'M': 11, 'dlalpha': 0.0, 'vlalpha': 0.3, 'lam': 2.0, 'beta': 0.02})
        elif dataset == "dataset_2":
            UBiRW = partial(UBiRW, A=a, SD=d, SV=v, **{
                            'alpha': 0.001, 'l1': 31, 'l2': 1, 'dlalpha': 0.5, 'vlalpha': 0.5, 'lam': 2.5})
            BiRW = partial(BiRW, A=a, SD=d, SV=v, **{'alpha': 0.6, 'M': 31, 'dlalpha': 0.6, 'vlalpha': 0.8999999999999999, 'lam': 1.5, 'beta': 0.04})
        else:
            UBiRW=partial(UBiRW, A=a, SD=d, SV=v, **{
                          'alpha': 0.001, 'l1': 11, 'l2': 1, 'dlalpha': 0.1, 'vlalpha': 0.1, 'lam': 2.5})
            BiRW=partial(BiRW, A=a, SD=d, SV=v, **{
                            'alpha': 0.6, 'M': 21, 'dlalpha': 0.3, 'vlalpha': 0.3, 'lam': 2.5, 'beta': 0.06})

        rr=pd.DataFrame(
            columns=["ACC", "Precision", "Recall", "Specifity", "AUC", "AUPR", "f1 score"])
        for i in range(1, 4):
            cvt="cv{}".format(i)
            print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            ro=train(UBiRW, cvt).rename("cv{}".format(i))
            rr=rr.append(ro)
        rr.to_excel("UBIRW_" + dataset + ".xlsx")
