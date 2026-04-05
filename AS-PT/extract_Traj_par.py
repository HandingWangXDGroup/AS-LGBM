from modcma import ModularCMAES
from modcma import Parameters
import nevergrad as ng
from multiprocessing import Process
import numpy as np
import csv
import Generation
import warnings
from ioh import get_problem, ProblemClass
from ma_bbob import ManyAffine
import ioh
import pandas as pd
import subprocess

D=10
def all_trajectories(func,lower=-10,upper=10):

    global current_trajectory
    def new_func(x):

        y = func(np.array([x]))
        current_trajectory.append(y[0])
        return y

    current_trajectory = []

    params=Parameters(d=D,lb=np.array([lower]*D),ub=np.array([upper]*D),bound_correction="saturate",budget=70)

    cma = ModularCMAES(new_func, D, budget=70,parameters=params)
    cma = cma.run()

    parametrization = ng.p.Instrumentation(x=ng.p.Array(shape=(10,), lower=lower, upper=upper))
    optimizer = ng.optimizers.DE(parametrization=parametrization, budget=210)
    optimizer.minimize(new_func)

    optimizer = ng.optimizers.PSO(parametrization=parametrization, budget=280)
    optimizer.minimize(new_func)

    return np.array(current_trajectory)

# def extract_Traj_RGI(ID):
#     le=np.loadtxt(open(r"D:\Pythonnnnnn\python\MyWork\EXP\MyDataset\Dataset_Expr\Expr-{}.csv".format(ID+1), "r"), delimiter=",")
#
#     with open(r'D:\Pythonnnnnn\python\MyWork\EXP\ComparativeData\PT\RGI_Traj-{}.csv'.format(ID + 1), 'a+',
#               newline='') as csvfile:
#         csvwriter = csv.writer(csvfile)
#         warnings.filterwarnings("ignore")
#         for i in range(len(le)):
#             exp = le[i, 1:]
#             ID = le[i, 0]
#             try:
#
#                 exp = exp[np.isnan(exp) == False]
#
#                 f = Generation.expr2func(exp, 10, 1)
#
#                 func = eval("lambda x: " + f)
#
#                 Traj=all_trajectories(func)
#                 csvwriter.writerow([ID] + Traj.tolist())
#                 csvfile.flush()
#             except:
#                 csvwriter.writerow([ID] + np.zeros([560]).tolist())
#                 csvfile.flush()
#                 continue

def extract_Traj_BBOB(ID):
    ID=ID+1
    with open(r'D:\Pythonnnnnn\python\MyWork\EXP\ComparativeData\PT\BBOB_Traj-{}.csv'.format(ID), 'a+',
              newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        warnings.filterwarnings("ignore")
        for i in range(100):
            problem = get_problem(ID, i, D, problem_class=ProblemClass.BBOB)
            func = lambda x: np.array(problem(x))
            try:
                Traj=all_trajectories(func,-5,5)-problem.optimum.y
                csvwriter.writerow(Traj.tolist())
                csvfile.flush()
            except:
                csvwriter.writerow(np.zeros([560]).tolist())
                csvfile.flush()
                continue

def extract_Traj_Affine(ID):
    ID=ID+1
    with open(r'D:\Pythonnnnnn\python\MyWork\EXP\ComparativeData\PT\Affine_Traj-{}.csv'.format(ID), 'a+',
              newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        warnings.filterwarnings("ignore")
        for i in range(1, 25):
            for alpha in np.linspace(0, 1, 21):
                if i == ID:
                    continue
                if alpha == 0 or alpha == 1:
                    continue
                if i < ID:
                    continue
                else:
                    pass
                problem1 = get_problem(ID, 0, D, problem_class=ProblemClass.BBOB)
                problem2 = get_problem(i, 0, D, problem_class=ProblemClass.BBOB)
                func = lambda x: (1 - alpha) * np.array(problem1(x)) + alpha * np.array(
                    problem2(x - problem1.optimum.x + problem2.optimum.x))
                try:
                    Traj=all_trajectories(func,-5,5)-(1 - alpha) *problem1.optimum.y-alpha*problem2.optimum.y
                    csvwriter.writerow(Traj.tolist())
                    csvfile.flush()
                except:
                    csvwriter.writerow(np.zeros([560]).tolist())
                    csvfile.flush()
                    continue



def extract_Traj_Zigzag(ID):

    for Instance in range(5):
        with open(r'D:\Pythonnnnnn\python\MyWork\EXP\ComparativeData\PT\Zigzag_Traj-{}-{}.csv'.format(ID,Instance), 'a+',
                  newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            warnings.filterwarnings("ignore")
            for M in [0.1, 0.5, 0.9, 1]:
                for Lamb in [0.01, 0.1, 0.5, 0.9, 0.99]:
                    for K in [1, 2, 4, 8, 16]:
                        func, _ = Zigzag(K, M, Lamb, D, ID)
                        try:
                            Traj=all_trajectories(func,-100,100)
                            csvwriter.writerow(Traj.tolist())
                            csvfile.flush()
                        except:
                            csvwriter.writerow(np.zeros([560]).tolist())
                            csvfile.flush()
                            continue

def extract_Traj_MA(ID):
    def get_func(i):

        f_new = ManyAffine(weights[i],
                           iids[i],
                           opt_loc[i], D)

        ioh.wrap_problem(
            f_new,
            name="ma-bbob-{}".format(i),
            optimization_type=ioh.OptimizationType.MIN,
            lb=-5,
            ub=5,
            dimension=D
        )

        f = ioh.get_problem("ma-bbob-{}".format(i),dimension=D)

        return f
    weights = np.loadtxt('/MyWork/EXP/MA-BBOB/weights.csv', delimiter=',')
    iids = np.loadtxt('/MyWork/EXP/MA-BBOB/iids.csv', delimiter=',')
    opt_loc = np.loadtxt('/MyWork/EXP/MA-BBOB/opt_loc.csv', delimiter=',')

    with open(r'D:\Pythonnnnnn\python\MyWork\EXP\ComparativeData\PT\MA_Traj-{}.csv'.format(ID), 'a+',
              newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        warnings.filterwarnings("ignore")
        for i in range(ID*500,(ID+1)*500):
            f = get_func(i)
            func = lambda x: np.array(f(x))
            try:
                Traj=all_trajectories(func,-5,5)
                csvwriter.writerow(Traj.tolist())
                csvfile.flush()
            except:
                csvwriter.writerow(np.zeros([560]).tolist())
                csvfile.flush()
                continue

def extract_Traj_POP(ID):
    def f(i, x):
        P = instances[i]
        r = P[:10]
        sigma = P[10:].reshape([10, 10])
        if len(np.shape(x)) == 1:
            x = np.array([x])
        fitness = []
        for i in range(len(x)):
            xx = x[i]
            f1 = xx @ r.T
            f2 = xx @ sigma @ xx.T
            fitness.append(-10 * f1 + f2)

        return np.array(fitness)

    varMin = -1
    varMax = 1

    D = 10

    instances = np.loadtxt('/MyWork/EXP/POP/instances.csv', delimiter=',')

    with open(r'D:\Pythonnnnnn\python\MyWork\EXP\ComparativeData\PT\POP_Traj-{}.csv'.format(ID), 'a+',
              newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        warnings.filterwarnings("ignore")
        for i in range(ID*1000,(ID+1)*1000):
            func= lambda x: f(i,x)
            try:
                Traj=all_trajectories(func,-1,1)
                csvwriter.writerow(Traj.tolist())
                csvfile.flush()
            except:
                csvwriter.writerow(np.zeros([560]).tolist())
                csvfile.flush()
                continue

def extract_Traj_TSFP(ID):
    def f(i, x):
        k = 9
        p = instances[i][np.isnan(instances[i]) == False]
        T = len(p)
        pt = p[k:T]
        if len(np.shape(x)) == 1:
            x = [x]
        fitness = []
        for i in range(len(x)):
            xx = x[i]
            ptp = []
            for ts in range(k, T):
                ptp.append(np.sum(xx[0] + xx[1:] * p[ts - k:ts]))
            fitness.append(np.sum((pt - ptp) ** 2) / (T - k))
        return np.array(fitness)

    varMin = -1
    varMax = 1

    D = 10

    instances=pd.read_csv('/MyWork/EXP/TSFP/instances.csv', delimiter=',', header=None).to_numpy()

    with open(r'D:\Pythonnnnnn\python\MyWork\EXP\ComparativeData\PT\TSFP_Traj-{}.csv'.format(ID), 'a+',
              newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        warnings.filterwarnings("ignore")
        for i in range(ID*500,(ID+1)*500):
            func = lambda x: f(i, x)
            try:
                Traj=all_trajectories(func,-1,1)
                csvwriter.writerow(Traj.tolist())
                csvfile.flush()
            except:
                csvwriter.writerow(np.zeros([560]).tolist())
                csvfile.flush()
                continue

def extract_Traj_Ealain(ID):
    # Ealain
    def f(num_pixl, x):
        fitness = []
        for i in range(len(x)):
            command = r"D:\Pythonnnnnn\python\MyWork\EXP\Ealain_data\example_so.exe {} 5".format(num_pixl)
            for j in range(len(x[i]) // 2):
                command = command + r" {} {}".format(x[i, j * 2], x[i, j * 2 + 1])

            result = subprocess.run(command, shell=True, text=True, capture_output=True)
            fitness.append(1 - (eval(result.stdout.split('\n')[0]) / num_pixl ** 2))
        return np.array(fitness)

    varMin = 0
    varMax = 1
    D = 10

    with open(r'/MyWork/EXP/ComparativeData/PT/Ealain_Traj.csv', 'a+', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        warnings.filterwarnings("ignore")

        for i in range(200):
            func = lambda x: f(i, x)
            try:
                Traj=all_trajectories(func,varMin,varMax)
                csvwriter.writerow(Traj.tolist())
                csvfile.flush()
                print(i)
            except:
                csvwriter.writerow(np.zeros([560]).tolist())
                csvfile.flush()
                continue

if __name__ == '__main__':
    extract_Traj_Ealain(0)
    # process_list = []
    # for i in range(10,20):  # 开启5个子进程执行fun1函数
    #     p = Process(target=extract_Traj_TSFP, args=(i,))  # 实例化进程对象
    #     p.start()
    #     process_list.append(p)
    #
    # for p in process_list:
    #     p.join()