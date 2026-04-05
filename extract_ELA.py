import numpy as np
import pandas as pd
from multiprocessing import Process
from pflacco.sampling import create_initial_sample
from pflacco.classical_ela_features import *
from pflacco.misc_features import *
from pflacco.local_optima_network_features import compute_local_optima_network, calculate_lon_features

import csv
import Generation


def Cal_ELA(x, y):
    # ################## Classical ELA Features #####################
    # Compute an exemplary feature set from the convential ELA features of the R-package flacco
    ela_meta = calculate_ela_meta(x, y)

    ela_distr = calculate_ela_distribution(x, y)

    ela_level = calculate_ela_level(x, y)

    # ################## Dispersion Features #####################
    disp = calculate_dispersion(x, y)

    # ################## Information Content-Based Features #####################
    ic = calculate_information_content(x, y)

    # ################## Nearest Better Features #####################
    nbc = calculate_nbc(x, y)

    # ################## Principal Components Features #####################
    pca = calculate_pca(x, y)
    # Compute an exemplary feature set from the novel features which are not part of the R-package flacco yet.
    fdc = calculate_fitness_distance_correlation(x, y)

    ela = (list(ela_meta.values())[:-1] + list(ela_distr.values())[:-1] +
           list(ela_level.values())[:-1] +
           list(disp.values())[:-1] + list(ic.values())[:-1] + list(nbc.values())[:-1] + list(fdc.values())[:-1] +
           list(pca.values())[:-1])
    return ela

def GenerateDataset(idfile):
    # 数据读取
    Exprs = np.loadtxt(r'MyDataset\Dataset_Expr\Expr-{}.csv'.format(idfile+1), delimiter=',')
    lower_bound = -10
    upper_bound = 10
    dim = 10

    with open(r'MyDataset\Dataset_ELA\ELA-{}.csv'.format(idfile + 1), 'a+',
              newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        warnings.filterwarnings("ignore")
        for i in range(len(Exprs)):
            exp = Exprs[i, 1:]
            ID = Exprs[i, 0]
            try:
                exp = exp[np.isnan(exp) == False]
                f = Generation.expr2func(exp, dim, 1)
                f = eval("lambda x: " + f)

                # Create inital sample using latin hyper cube sampling
                X = create_initial_sample(dim, sample_type='lhs', lower_bound=lower_bound,
                                          upper_bound=upper_bound, )
                x = X.to_numpy()
                y = f(x)

                ela = Cal_ELA(x, y)

                csvwriter.writerow([ID]+ela)
                csvfile.flush()
            except:
                csvwriter.writerow([ID]+np.zeros([61]).tolist())
                csvfile.flush()
                continue

def debug():
    Exprs = np.loadtxt(r'MyDataset\Dataset_Expr\Expr-{}.csv'.format(0 + 1), delimiter=',')
    lower_bound = -10
    upper_bound = 10
    dim = 10

    for i in range(len(Exprs)):
        exp = Exprs[i, 1:]
        ID = Exprs[i, 0]

        exp = exp[np.isnan(exp) == False]
        f = Generation.expr2func(exp, dim, 1)
        f = eval("lambda x: " + f)

        # Create inital sample using latin hyper cube sampling
        X = create_initial_sample(dim, sample_type='lhs', lower_bound=lower_bound,
                                  upper_bound=upper_bound,sample_coefficient=20 )
        x = X.to_numpy()
        y = f(x)

        ela = Cal_ELA(x, y)

        print(ela)



if __name__ == '__main__':
    # process_list = []
    # for i in range(3):  # 开启5个子进程执行fun1函数
    #     p = Process(target=GenerateDataset, args=(i,))  # 实例化进程对象
    #     p.start()
    #     process_list.append(p)
    #
    # for p in process_list:
    #     p.join()
    # X = create_initial_sample(10, sample_type='lhs', lower_bound=-10,
    #                           upper_bound=10,sample_coefficient=20 )
    # print(X)
    # print(len(X))
    debug()
    # X = create_initial_sample(10, sample_type='lhs', lower_bound=-10,
    #                           upper_bound=10, )
    #
    # np.savetxt('MyDataset//DoE.csv', X, delimiter=',')


