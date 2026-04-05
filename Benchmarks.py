import pandas as pd
import numpy as np

def get_dataset(ID):
    # ID: 0RGI,1BBOB,2Affine,3Zigzag,4GPB,5RGF
    if ID==0:
        # RGI
        RGI_ela = np.zeros([1, 61])
        RGI_rt = np.zeros([1, 300])
        for i in range(25):
            RGI_ela = np.append(RGI_ela, (pd.read_csv(
                "D:\Pythonnnnnn\python\MyWork\EXP\MyDataset\Dataset_ELA_500_DoE\ELA-{}.csv".format(i + 1), header=None,
                skip_blank_lines=False).fillna(0).to_numpy())[:, 1:], axis=0)
            RGI_rt = np.append(RGI_rt, (np.loadtxt(
                open("D:\Pythonnnnnn\python\MyWork\EXP\MyDataset\Dataset_RT\RT-{}.csv".format(i + 1), "r"),
                delimiter=","))[:, 1:], axis=0)
        # 筛选数据
        RGI_ela, RGI_rt = data_screening(RGI_ela, RGI_rt)
        RGI_ert, RGI_label = labeling(RGI_rt)
        return RGI_ela,RGI_label

    elif ID==1:
        # BBOB
        BBOB_ela = np.zeros([1, 61])
        BBOB_rt = np.zeros([1, 300])
        for i in range(24):
            BBOB_ela = np.append(BBOB_ela,
                                 np.loadtxt(open("D:\Dataset\Mywork\DataAll\BBOB\\bbob-{}.csv".format(i + 1), "r"),
                                            delimiter=",")[:, 600:], axis=0)
            BBOB_rt = np.append(BBOB_rt,
                                np.loadtxt(open("D:\Dataset\Mywork\DataAll\BBOB\\bbob-{}.csv".format(i + 1), "r"),
                                           delimiter=",")[:, 300:600], axis=0)

        BBOB_ela, BBOB_rt = data_screening(BBOB_ela, BBOB_rt)
        BBOB_ert, BBOB_label = labeling(BBOB_rt)
        return BBOB_ela, BBOB_label

    elif ID==2:
        # BBOB_Affine
        Affine= np.loadtxt(open(r"D:\Dataset\Mywork\DataAll\Affine\affine.csv", "r"), delimiter=",")
        Affine_ela = Affine[:, 600:]
        Affine_rt = Affine[:, 300:600]

        Affine_ela, Affine_rt = data_screening(Affine_ela, Affine_rt)
        Affine_ert, Affine_label = labeling(Affine_rt)
        return Affine_ela, Affine_label
    elif ID == 3:
        # Zigzag
        Zigzag_ela = np.zeros([1, 61])
        Zigzag_rt = np.zeros([1, 300])
        for i in range(5):
            for j in range(5):
                Zigzag_ela = np.append(Zigzag_ela, np.loadtxt(
                    open("D:\Dataset\Mywork\DataAll\Zigzag\\zigzag-{}-{}.csv".format(i, j), "r"), delimiter=",")[:,
                                                   600:], axis=0)
                Zigzag_rt = np.append(Zigzag_rt, np.loadtxt(
                    open("D:\Dataset\Mywork\DataAll\Zigzag\\zigzag-{}-{}.csv".format(i, j), "r"), delimiter=",")[:,
                                                 300:600], axis=0)

        Zigzag_ela, Zigzag_rt = data_screening(Zigzag_ela, Zigzag_rt)
        Zigzag_ert, Zigzag_label = labeling(Zigzag_rt)
        return Zigzag_ela,Zigzag_label

    elif ID == 4:
        # GPB
        GPB_ela = np.zeros([1, 61])
        GPB_rt = np.zeros([1, 300])
        for i in range(5):
            GPB_ela = np.append(GPB_ela, np.loadtxt(open("D:\Dataset\Mywork\DataAll\GPB\\lfA-{}.csv".format(i), "r"),
                                                    delimiter=",")[:, 600:], axis=0)
            GPB_rt = np.append(GPB_rt, np.loadtxt(open("D:\Dataset\Mywork\DataAll\GPB\\lfA-{}.csv".format(i), "r"),
                                                  delimiter=",")[:, 300:600], axis=0)
        for i in range(5):
            GPB_ela = np.append(GPB_ela, np.loadtxt(open("D:\Dataset\Mywork\DataAll\GPB\\lfB-{}.csv".format(i), "r"),
                                                    delimiter=",")[:, 600:], axis=0)
            GPB_rt = np.append(GPB_rt, np.loadtxt(open("D:\Dataset\Mywork\DataAll\GPB\\lfB-{}.csv".format(i), "r"),
                                                  delimiter=",")[:, 300:600], axis=0)

        GPB_ela, GPB_rt = data_screening(GPB_ela, GPB_rt)
        GPB_ert, GPB_label = labeling(GPB_rt)
        return GPB_ela,GPB_label

    elif ID == 5:
        # RGF
        RGF_ela = np.zeros([1, 61])
        RGF_label = np.zeros([1])
        for i in range(20):
            RGF_ela = np.append(RGF_ela, np.loadtxt(
                open("D:\Dataset\Mywork\DataAll\RGF\RGF_ELA\lf-{}.csv".format(i + 1), "r"), delimiter=",")[:, 1:],
                                axis=0)
            RGF_label = np.append(RGF_label, np.loadtxt(
                open("D:\Dataset\Mywork\DataAll\RGF\RGF_ELA\lf-{}.csv".format(i + 1), "r"), delimiter=",")[:, 0],
                                  axis=0)

        a = np.sum(RGF_ela, axis=1) != 0
        b = np.sum(np.isnan(RGF_ela), axis=1) == 0
        c = np.sum(np.isinf(RGF_ela), axis=1) == 0
        d = np.sum(np.abs(RGF_ela) > 10 ** 8, axis=1) == 0
        ind = a & b & c & d
        RGF_ela = RGF_ela[ind, :]
        RGF_label = RGF_label[ind]
        return RGF_ela, RGF_label.reshape([-1, 1])

    elif ID == 6:
        # MA-BBOB
        BBOB_ela = np.zeros([1, 61])
        BBOB_rt = np.zeros([1, 300])
        for i in range(20):
            BBOB_ela = np.append(BBOB_ela,
                                 np.loadtxt(open("D:\Pythonnnnnn\python\MyWork\EXP\MA-BBOB\MA-BBOB\\MA-BBOB-{}.csv".format(i), "r"),
                                            delimiter=",")[:, 601:], axis=0)
            BBOB_rt = np.append(BBOB_rt,
                                np.loadtxt(open("D:\Pythonnnnnn\python\MyWork\EXP\MA-BBOB\MA-BBOB\\MA-BBOB-{}.csv".format(i), "r"),
                                           delimiter=",")[:, 301:601], axis=0)

        BBOB_ela, BBOB_rt = data_screening(BBOB_ela, BBOB_rt)
        BBOB_ert, BBOB_label = labeling(BBOB_rt)
        return BBOB_ela, BBOB_label

    elif ID==7:
        # POP
        POP_ela = np.zeros([1, 61])
        POP_rt = np.zeros([1, 300])
        for i in range(10):
            POP_ela = np.append(POP_ela,
                                 np.loadtxt(open("D:\Pythonnnnnn\python\MyWork\EXP\POP\Dataset_ELA\\ela-{}.csv".format(i+1), "r"),
                                            delimiter=",")[:, 1:], axis=0)
            POP_rt = np.append(POP_rt,
                                np.loadtxt(open("D:\Pythonnnnnn\python\MyWork\EXP\POP\Dataset_RT\\RT-{}.csv".format(i+1), "r"),
                                           delimiter=",")[:, 1:], axis=0)

        POP_ela, POP_rt = data_screening(POP_ela, POP_rt)
        POP_ert, POP_label = labeling(POP_rt)
        return POP_ela, POP_label
    elif ID==8:
        # tsfp
        POP_ela = np.zeros([1, 61])
        POP_rt = np.zeros([1, 300])
        for i in range(20):
            POP_ela = np.append(POP_ela,
                                 np.loadtxt(open("D:\Pythonnnnnn\python\MyWork\EXP\TSFP\Dataset_ELA\\ela-{}.csv".format(i+1), "r"),
                                            delimiter=",")[:, 1:], axis=0)
            POP_rt = np.append(POP_rt,
                                np.loadtxt(open("D:\Pythonnnnnn\python\MyWork\EXP\TSFP\Dataset_RT\\RT-{}.csv".format(i+1), "r"),
                                           delimiter=",")[:, 1:], axis=0)

        POP_ela, POP_rt = data_screening(POP_ela, POP_rt)
        POP_ert, POP_label = labeling(POP_rt)
        return POP_ela, POP_label
    elif ID==9:
        # Ealain
        ela = np.loadtxt(r'D:\Pythonnnnnn\python\MyWork\EXP\Ealain\ELA.csv',delimiter=',')
        rt = np.loadtxt(r'D:\Pythonnnnnn\python\MyWork\EXP\Ealain\RT.csv',delimiter=',')

        ela, lRT = data_screening(ela, rt)

        lERT = np.zeros([len(lRT), 10])
        for i in range(len(lRT)):
            for j in range(10):
                lERT[i, j] = (np.sum(lRT[i, j * 10:(j + 1) * 10]) + np.sum(
                    lRT[i, j * 10:(j + 1) * 10] == -1) * 10001) / np.max(
                    [np.sum(lRT[i, j * 10:(j + 1) * 10] != -1), 1])
        y = np.argmin(lERT, axis=1)
        label = np.zeros([len(y), 101])
        label[:, 0] = y
        lRT[lRT == -1] = 10000
        label[:, 1:] = lRT
        return ela, label



def labeling(lRT):
    lERT = cal_ert(lRT)
    y = np.argmin(lERT, axis=1)
    label = np.zeros([len(y), 301])
    label[:, 0] = y
    lRT[lRT == -1] = 10000
    label[:, 1:] = lRT
    return lERT, label

def data_screening(ela,rt=0):
    a=np.sum(ela,axis=1)!=0
    b=np.sum(np.isnan(ela),axis=1)==0
    c=np.sum(np.isinf(ela),axis=1)==0
    d=np.sum(np.abs(ela)>10**8,axis=1)==0
    e=np.sum(rt,axis=1)!=0
    ind=a & b & c & d & e
    return ela[ind,:],rt[ind,:]

def cal_ert(lRT):
    lERT=np.zeros([len(lRT),10])
    for i in range(len(lRT)):
        for j in range(10):
            lERT[i,j]=(np.sum(lRT[i,j*30:(j+1)*30])+np.sum(lRT[i,j*30:(j+1)*30]==-1)*10001)/np.max([np.sum(lRT[i,j*30:(j+1)*30]!=-1),1])
    return lERT

def get_RGI(ID):
    # ID: 0RGI,1BBOB,2Affine,3Zigzag,4GPB,5RGF
    if ID==0:
        # RGI
        RGI_ela = np.zeros([1, 61])
        RGI_rt = np.zeros([1, 300])
        for i in range(25):
            RGI_ela = np.append(RGI_ela, (pd.read_csv(
                "D:\Pythonnnnnn\python\MyWork\EXP\MyDataset\Dataset_ELA_200\ELA-{}.csv".format(i + 1), header=None,
                skip_blank_lines=False).fillna(0).to_numpy())[:, 1:], axis=0)
            RGI_rt = np.append(RGI_rt, (np.loadtxt(
                open("D:\Pythonnnnnn\python\MyWork\EXP\MyDataset\Dataset_RT\RT-{}.csv".format(i + 1), "r"),
                delimiter=","))[:, 1:], axis=0)
        # 筛选数据
        RGI_ela, RGI_rt = data_screening(RGI_ela, RGI_rt)
        RGI_ert, RGI_label = labeling(RGI_rt)
        return RGI_ela,RGI_label
    elif ID==1:
        # RGI
        RGI_ela = np.zeros([1, 61])
        RGI_rt = np.zeros([1, 300])
        for i in range(25):
            RGI_ela = np.append(RGI_ela, (pd.read_csv(
                "D:\Pythonnnnnn\python\MyWork\EXP\MyDataset\Dataset_ELA_300\ELA-{}.csv".format(i + 1), header=None,
                skip_blank_lines=False).fillna(0).to_numpy())[:, 1:], axis=0)
            RGI_rt = np.append(RGI_rt, (np.loadtxt(
                open("D:\Pythonnnnnn\python\MyWork\EXP\MyDataset\Dataset_RT\RT-{}.csv".format(i + 1), "r"),
                delimiter=","))[:, 1:], axis=0)
        # 筛选数据
        RGI_ela, RGI_rt = data_screening(RGI_ela, RGI_rt)
        RGI_ert, RGI_label = labeling(RGI_rt)
        return RGI_ela,RGI_label
    elif ID==2:
        # RGI
        RGI_ela = np.zeros([1, 61])
        RGI_rt = np.zeros([1, 300])
        for i in range(25):
            RGI_ela = np.append(RGI_ela, (pd.read_csv(
                "D:\Pythonnnnnn\python\MyWork\EXP\MyDataset\Dataset_ELA\ELA-{}.csv".format(i + 1), header=None,
                skip_blank_lines=False).fillna(0).to_numpy())[:, 1:], axis=0)
            RGI_rt = np.append(RGI_rt, (np.loadtxt(
                open("D:\Pythonnnnnn\python\MyWork\EXP\MyDataset\Dataset_RT\RT-{}.csv".format(i + 1), "r"),
                delimiter=","))[:, 1:], axis=0)
        # 筛选数据
        RGI_ela, RGI_rt = data_screening(RGI_ela, RGI_rt)
        RGI_ert, RGI_label = labeling(RGI_rt)
        return RGI_ela,RGI_label
    elif ID==3:
        # RGI
        RGI_ela = np.zeros([1, 61])
        RGI_rt = np.zeros([1, 300])
        for i in range(25):
            RGI_ela = np.append(RGI_ela, (pd.read_csv(
                "D:\Pythonnnnnn\python\MyWork\EXP\MyDataset\Dataset_ELA_800\ELA-{}.csv".format(i + 1), header=None,
                skip_blank_lines=False).fillna(0).to_numpy())[:, 1:], axis=0)
            RGI_rt = np.append(RGI_rt, (np.loadtxt(
                open("D:\Pythonnnnnn\python\MyWork\EXP\MyDataset\Dataset_RT\RT-{}.csv".format(i + 1), "r"),
                delimiter=","))[:, 1:], axis=0)
        # 筛选数据
        RGI_ela, RGI_rt = data_screening(RGI_ela, RGI_rt)
        RGI_ert, RGI_label = labeling(RGI_rt)
        return RGI_ela,RGI_label
    elif ID==4:
        # RGI
        RGI_ela = np.zeros([1, 61])
        RGI_rt = np.zeros([1, 300])
        for i in range(25):
            RGI_ela = np.append(RGI_ela, (pd.read_csv(
                "D:\Pythonnnnnn\python\MyWork\EXP\MyDataset\Dataset_ELA_200_DoE\ELA-{}.csv".format(i + 1), header=None,
                skip_blank_lines=False).fillna(0).to_numpy())[:, 1:], axis=0)
            RGI_rt = np.append(RGI_rt, (np.loadtxt(
                open("D:\Pythonnnnnn\python\MyWork\EXP\MyDataset\Dataset_RT\RT-{}.csv".format(i + 1), "r"),
                delimiter=","))[:, 1:], axis=0)
        # 筛选数据
        RGI_ela, RGI_rt = data_screening(RGI_ela, RGI_rt)
        RGI_ert, RGI_label = labeling(RGI_rt)
        return RGI_ela,RGI_label

    elif ID==5:
        # RGI
        RGI_ela = np.zeros([1, 61])
        RGI_rt = np.zeros([1, 300])
        for i in range(25):
            RGI_ela = np.append(RGI_ela, (pd.read_csv(
                "D:\Pythonnnnnn\python\MyWork\EXP\MyDataset\Dataset_ELA_300_DoE\ELA-{}.csv".format(i + 1), header=None,
                skip_blank_lines=False).fillna(0).to_numpy())[:, 1:], axis=0)
            RGI_rt = np.append(RGI_rt, (np.loadtxt(
                open("D:\Pythonnnnnn\python\MyWork\EXP\MyDataset\Dataset_RT\RT-{}.csv".format(i + 1), "r"),
                delimiter=","))[:, 1:], axis=0)
        # 筛选数据
        RGI_ela, RGI_rt = data_screening(RGI_ela, RGI_rt)
        RGI_ert, RGI_label = labeling(RGI_rt)
        return RGI_ela,RGI_label

    elif ID==6:
        # RGI
        RGI_ela = np.zeros([1, 61])
        RGI_rt = np.zeros([1, 300])
        for i in range(25):
            RGI_ela = np.append(RGI_ela, (pd.read_csv(
                "D:\Pythonnnnnn\python\MyWork\EXP\MyDataset\Dataset_ELA_500_DoE\ELA-{}.csv".format(i + 1), header=None,
                skip_blank_lines=False).fillna(0).to_numpy())[:, 1:], axis=0)
            RGI_rt = np.append(RGI_rt, (np.loadtxt(
                open("D:\Pythonnnnnn\python\MyWork\EXP\MyDataset\Dataset_RT_500_DoE\RT-{}.csv".format(i + 1), "r"),
                delimiter=","))[:, 1:], axis=0)
        # 筛选数据
        RGI_ela, RGI_rt = data_screening(RGI_ela, RGI_rt)
        RGI_ert, RGI_label = labeling(RGI_rt)
        return RGI_ela,RGI_label

    elif ID==7:
        # RGI
        RGI_ela = np.zeros([1, 61])
        RGI_rt = np.zeros([1, 300])
        for i in range(25):
            RGI_ela = np.append(RGI_ela, (pd.read_csv(
                "D:\Pythonnnnnn\python\MyWork\EXP\MyDataset\Dataset_ELA_800_DoE\ELA-{}.csv".format(i + 1), header=None,
                skip_blank_lines=False).fillna(0).to_numpy())[:, 1:], axis=0)
            RGI_rt = np.append(RGI_rt, (np.loadtxt(
                open("D:\Pythonnnnnn\python\MyWork\EXP\MyDataset\Dataset_RT\RT-{}.csv".format(i + 1), "r"),
                delimiter=","))[:, 1:], axis=0)
        # 筛选数据
        RGI_ela, RGI_rt = data_screening(RGI_ela, RGI_rt)
        RGI_ert, RGI_label = labeling(RGI_rt)
        return RGI_ela,RGI_label
if __name__=='__main__':
    get_dataset(0)


