import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from doe2vec import doe_model
import subprocess
import numpy as np
import Generation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import scipy.stats as stats
from multiprocessing import Process
import csv
from ioh import get_problem, ProblemClass
from Zigzag import Zigzag
import warnings
import ioh
from ma_bbob import ManyAffine

# 加载VAE预训练模型
obj = doe_model(
            10,
            9,
            n= 250000,
            latent_dim=32,
            kl_weight=0.001,
            use_mlflow=False,
            model_type="VAE"
        )
# obj.load_from_huggingface()
obj.loadModel('../VAE')
samples=obj.sample

# # #RGI
# with open(r'D:\Pythonnnnnn\python\MyWork\EXP\ComparativeData\D2Vae\RGI_VAE.csv', 'a+',newline='') as csvfile:
#     le=np.zeros([1, 250])
#     for i in range(25):
#         le_=np.loadtxt(open(r"D:\Pythonnnnnn\python\MyWork\EXP\MyDataset\Dataset_Expr\Expr-{}.csv".format(i+1), "r"), delimiter=",")
#         le= np.append(le, np.append(le_,np.zeros([le_.shape[0],250-le_.shape[1]]),axis=1), axis=0)
#     csvwriter = csv.writer(csvfile)
#     warnings.filterwarnings("ignore")
#     samples=samples*20-10
#     for i in range(len(le)):
#         exp = le[i, 1:]
#         ID = le[i, 0]
#         try:
#             exp = exp[np.isnan(exp) == False]
#
#             f = Generation.expr2func(exp, 10, 1)
#
#             f = eval("lambda x: " + f)
#             fx = f(samples)
#             fx = (fx - np.min(fx)) / (np.max(fx) - np.min(fx))
#             encoded = obj.encode([fx])
#             print(i)
#             csvwriter.writerow([ID] + encoded.tolist()[0])
#             csvfile.flush()
#         except:
#             csvwriter.writerow([ID] + np.zeros([32]).tolist())
#             csvfile.flush()

# # #BBOB
# with open(r'D:\Pythonnnnnn\python\MyWork\EXP\ComparativeData\D2Vae\BBOB_VAE.csv', 'a+',newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     warnings.filterwarnings("ignore")
#     samples = samples * 10 - 5
#     for ID in range(24):
#         for i in range(100):
#             try:
#                 problem = get_problem(ID+1, i, 10, problem_class=ProblemClass.BBOB)
#                 f = lambda x: problem(x)
#
#                 fx = f(samples)
#                 fx = (fx - np.min(fx)) / (np.max(fx) - np.min(fx))
#                 encoded = obj.encode([fx])
#                 print(i)
#                 csvwriter.writerow( encoded.tolist()[0])
#                 csvfile.flush()
#             except:
#                 csvwriter.writerow(np.zeros([32]).tolist())
#                 csvfile.flush()

# # #BBOB-Affine
# with open(r'D:\Pythonnnnnn\python\MyWork\EXP\ComparativeData\D2Vae\Affine_VAE.csv', 'a+',newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     warnings.filterwarnings("ignore")
#     samples = samples * 10 - 5
#     D=10
#     for ID in range(1, 25):
#         for i in range(1, 25):
#             for alpha in np.linspace(0, 1, 21):
#                 if i == ID:
#                     continue
#                 if alpha == 0 or alpha == 1:
#                     continue
#                 if i < ID:
#                     continue
#                 else:
#                     problem1 = get_problem(ID, 0, D, problem_class=ProblemClass.BBOB)
#                     problem2 = get_problem(i, 0, D, problem_class=ProblemClass.BBOB)
#                     func = lambda x: (1 - alpha) * np.array(problem1(x)) + alpha * np.array(
#                         problem2(x - problem1.optimum.x + problem2.optimum.x))
#
#                     fx = func(samples)
#                     fx = (fx - np.min(fx)) / (np.max(fx) - np.min(fx))
#                     encoded = obj.encode([fx])
#                     print(i)
#                     csvwriter.writerow(encoded.tolist()[0])
#                     csvfile.flush()

# # Zigzag
# with open(r'D:\Pythonnnnnn\python\MyWork\EXP\ComparativeData\D2Vae\Zigzag_VAE.csv', 'a+',newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     warnings.filterwarnings("ignore")
#     samples = samples * 200 - 100
#     for ID in range(5):
#         for Instance in range(5):
#             for M in [0.1, 0.5, 0.9, 1]:
#                 for Lamb in [0.01, 0.1, 0.5, 0.9, 0.99]:
#                     for K in [1, 2, 4, 8, 16]:
#                         # try:
#                         func, _ = Zigzag(K, M, Lamb, 10, ID, Instance)
#                         fx = func(samples)
#                         fx = (fx - np.min(fx)) / (np.max(fx) - np.min(fx))
#                         encoded = obj.encode([fx])
#                         csvwriter.writerow( encoded.tolist()[0])
#                         csvfile.flush()
#                         # except:
#                         #     csvwriter.writerow(np.zeros([32]).tolist())
#                         #     csvfile.flush()
#

#MA-BBOB
# def get_func(i):
#     f_new = ManyAffine(weights[i],
#                        iids[i],
#                        opt_loc[i], 10)
#
#     ioh.wrap_problem(
#         f_new,
#         name="ma-bbob-{}".format(i),
#         optimization_type=ioh.OptimizationType.MIN,
#         lb=-5,
#         ub=5,
#         dimension=10
#     )
#
#     f = ioh.get_problem("ma-bbob-{}".format(i), dimension=10)
#
#     return f
#
#
# weights = np.loadtxt('D:\Pythonnnnnn\python\MyWork\EXP\MA-BBOB\weights.csv', delimiter=',')
# iids = np.loadtxt('D:\Pythonnnnnn\python\MyWork\EXP\MA-BBOB\iids.csv', delimiter=',')
# opt_loc = np.loadtxt('D:\Pythonnnnnn\python\MyWork\EXP\MA-BBOB\opt_loc.csv', delimiter=',')
# with open(r'D:\Pythonnnnnn\python\MyWork\EXP\ComparativeData\D2Vae\MA_VAE.csv', 'a+',newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     warnings.filterwarnings("ignore")
#     samples = samples * 10 - 5
#     for i in range(10000):
#         try:
#             func = get_func(i)
#             f = lambda x: np.array(func(x))
#
#             fx = f(samples)
#             fx = (fx - np.min(fx)) / (np.max(fx) - np.min(fx))
#             encoded = obj.encode([fx])
#             print(i)
#             csvwriter.writerow(encoded.tolist()[0])
#             csvfile.flush()
#         except:
#             csvwriter.writerow(np.zeros([32]).tolist())
#             csvfile.flush()


# POP
# def f(i, x):
#     P = instances[i]
#     r = P[:10]
#     sigma = P[10:].reshape([10, 10])
#     if len(np.shape(x)) == 1:
#         x = np.array([x])
#     fitness = []
#     for i in range(len(x)):
#         xx = x[i]
#         f1 = xx @ r.T
#         f2 = xx @ sigma @ xx.T
#         fitness.append(-10 * f1 + f2)
#
#     return np.array(fitness)
#
#
# varMin = -1
# varMax = 1
#
# D = 10
#
# instances = np.loadtxt('D:\Pythonnnnnn\python\MyWork\EXP\POP\instances.csv', delimiter=',')
#
# with open(r'D:\Pythonnnnnn\python\MyWork\EXP\ComparativeData\D2Vae\POP_VAE.csv', 'a+',newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     warnings.filterwarnings("ignore")
#     samples = samples * 2 - 1
#     for i in range(10000):
#         try:
#             func= lambda x: f(i,x)
#
#             fx = func(samples)
#             fx = (fx - np.min(fx)) / (np.max(fx) - np.min(fx))
#             encoded = obj.encode([fx])
#             print(i)
#             csvwriter.writerow(encoded.tolist()[0])
#             csvfile.flush()
#         except:
#             csvwriter.writerow(np.zeros([32]).tolist())
#             csvfile.flush()


#TSFP
# def f(i, x):
#     k = 9
#     p = instances[i][np.isnan(instances[i]) == False]
#     T = len(p)
#     pt = p[k:T]
#     if len(np.shape(x)) == 1:
#         x = [x]
#     fitness = []
#     for i in range(len(x)):
#         xx = x[i]
#         ptp = []
#         for ts in range(k, T):
#             ptp.append(np.sum(xx[0] + xx[1:] * p[ts - k:ts]))
#         fitness.append(np.sum((pt - ptp) ** 2) / (T - k))
#     return np.array(fitness)
#
#
# varMin = -1
# varMax = 1
#
# D = 10
#
# instances = pd.read_csv('D:\Pythonnnnnn\python\MyWork\EXP\TSFP\instances.csv', delimiter=',', header=None).to_numpy()
#
# with open(r'D:\Pythonnnnnn\python\MyWork\EXP\ComparativeData\D2Vae\TSFP_VAE.csv', 'a+',newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     warnings.filterwarnings("ignore")
#     samples = samples * 2 - 1
#     for i in range(10000):
#         try:
#             func = lambda x: f(i, x)
#
#             fx = func(samples)
#             fx = (fx - np.min(fx)) / (np.max(fx) - np.min(fx))
#             encoded = obj.encode([fx])
#             print(i)
#             csvwriter.writerow(encoded.tolist()[0])
#             csvfile.flush()
#         except:
#             csvwriter.writerow(np.zeros([32]).tolist())
#             csvfile.flush()


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
#
with open(r'D:\Pythonnnnnn\python\MyWork\EXP\ComparativeData\D2Vae\Ealain_VAE.csv', 'a+',newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    warnings.filterwarnings("ignore")
    for i in range(200):
        try:
            func = lambda x: f(i, x)

            fx = func(samples)
            fx = (fx - np.min(fx)) / (np.max(fx) - np.min(fx))
            encoded = obj.encode([fx])
            print(i)
            csvwriter.writerow(encoded.tolist()[0])
            csvfile.flush()
        except:
            csvwriter.writerow(np.zeros([32]).tolist())
            csvfile.flush()