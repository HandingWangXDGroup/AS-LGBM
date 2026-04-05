import random
import numpy as np
import time
# from Generation import expr2func


def ABC(func, nVar,VarMin,VarMax, nPop, MaxIt,S=0,His=0):
    varSize = [1, nVar]
    # 参数配置
    a = 1  # 加速度系数
    nLooker = nPop  # 引领蜂个数
    L = round(0.6 * nVar * nPop)  # 侦察蜂阈值

    # 初始化
    if isinstance(S,np.ndarray):
        ind=func(S).argsort()[:nPop]
        PopPosition=S[ind].copy()
    else:
        PopPosition = np.random.uniform(VarMin, VarMax, [nPop, nVar])  # 蜜源初始化
    PopFitness = func(PopPosition)

    BestPosition = PopPosition[np.argmin(PopFitness)].copy()  # 全局最优解
    BestFitness = min(PopFitness)  # 全局最优值
    HistFitness = np.zeros(MaxIt//2+1)  # 记录最优
    HistFitness[0]=BestFitness
    C = np.zeros([nPop, 1])  # 放弃蜜源计时器
    ind = np.arange(nPop)  # 用于辅助的索引

    # 开始搜索
    start = time.time()
    for It in range(MaxIt // 2):
        # 引领蜂
        k = np.random.randint(0, nPop, nPop)
        k[k == ind] = k[k == ind] + 1  # 使得i!=k
        k[nPop - 1] = np.random.randint(0, nPop - 1)

        phi = a * np.random.uniform(-1, 1, [nPop, nVar])  # 定义加速度系数
        NewPositon = PopPosition + phi * (PopPosition - PopPosition[k])  # 搜索到新的蜜源
        NewPositon[NewPositon > VarMax] = VarMax  # 边界处理
        NewPositon[NewPositon < VarMin] = VarMin

        NewFitness = func(NewPositon)  # 计算搜索到的蜜源的适应值

        index = NewFitness < PopFitness  # 贪婪比较，更新蜜源
        PopPosition[index] = NewPositon[index].copy()
        PopFitness[index] = NewFitness[index]
        C[~index] = C[~index] + 1  # 放弃蜜源计数

        # 跟随蜂
        wheel = np.cumsum(
            np.exp(-PopFitness / np.abs(np.mean(PopFitness))) / sum(
                np.exp(-PopFitness / np.abs(np.mean(PopFitness)))))  # 轮盘赌
        # print(wheel)
        for i in range(nLooker):
            try:
                follow_index = np.argwhere(wheel > np.random.random())[0]
            except:
                follow_index = np.random.randint(0, nPop)


            k = np.random.randint(0, nPop)
            phi = a * np.random.uniform(-1, 1, varSize)  # 定义加速度系数
            NewPositon = PopPosition[follow_index] + phi * (PopPosition[follow_index] - PopPosition[k])  # 搜索到新的蜜源
            NewPositon[NewPositon > VarMax] = VarMax  # 边界处理
            NewPositon[NewPositon < VarMin] = VarMin

            NewFitness = func(NewPositon)  # 计算搜索到的蜜源的适应值

            if NewFitness < PopFitness[follow_index]:  # 贪婪比较，更新蜜源
                PopPosition[follow_index] = NewPositon.copy()
                PopFitness[follow_index] = NewFitness
            else:
                C[follow_index] = C[follow_index] + 1  # 放弃蜜源计数

        # 侦察蜂
        for i in range(nPop):
            if C[i] >= L:
                PopPosition[i] = np.random.uniform(VarMin, VarMax, varSize)
                PopFitness[i] = func(PopPosition[i].reshape(varSize))
                C[i] = 0

        # 更新全局最优
        if min(PopFitness) < BestFitness:
            BestPosition = PopPosition[np.argmin(PopFitness)].copy()  # 全局最优解
            BestFitness = min(PopFitness)  # 全局最优值
        HistFitness[It+1] = BestFitness

        # # 展示迭代信息
        # print("Iteration {}: Best func = {}".format(It, BestFitness))
    # # 运行结果
    # end = time.time()  # 运行结束时刻
    # print("最优解：{}".format(BestPosition))
    # print("最优值：{}".format(BestFitness))
    # print('运行时间：{}s'.format(end - start))
    # print(BestPosition)
    if His :
        return BestFitness,HistFitness
    else:
        return BestFitness


def ACO(func, nVar,VarMin,VarMax, nPop, MaxIt,S=0,His=0):
    varSize = [1, nVar]

    # 参数配置
    q = 0.5
    w = 1 / (np.sqrt(2 * np.pi) * q * nPop) * np.exp(-0.5 * ((np.arange(nPop)) / (q * nPop)) ** 2)
    p = w / np.sum(w)
    idx = np.arange(2 * nPop)

    # 种群初始化
    if isinstance(S,np.ndarray):
        ind=func(S).argsort()[:nPop]
        Pop=S[ind].copy()
    else:
        Pop = np.random.uniform(VarMin, VarMax, [nPop, nVar])  # 蚂蚁初始化
    PopFitness = func(Pop)
    rank = idx[PopFitness.argsort()][:nPop]
    Pop = Pop[rank]
    PopFitness = PopFitness[rank]
    BestPop = Pop[0].copy()  # 全局最优
    BestFitness = PopFitness[0]  # 全局最优值
    HistFitness = np.zeros(MaxIt+1)  # 记录最优
    HistFitness[0] = BestFitness

    # 开始搜索
    start = time.time()
    for It in range(MaxIt):
        Mean = Pop.copy()
        Std = np.zeros([nPop, nVar])
        for i in range(nPop):
            Std[i] = np.sum(abs(Pop[i] - Pop), axis=0) / (nPop - 1)
        newPop = np.zeros([nPop, nVar])
        i = np.random.choice(np.arange(nPop), [nPop, nVar], p=p)
        for j in range(nVar):
            newPop[:, j] = Mean[i[:, j], j] + Std[i[:, j], j] * np.random.randn(nPop)
        newPop[newPop > VarMax] = VarMax
        newPop[newPop < VarMin] = VarMin

        Pop = np.vstack((Pop, newPop))
        PopFitness = np.vstack((PopFitness, func(newPop))).reshape([-1])
        rank = idx[PopFitness.argsort()][:nPop]
        Pop = Pop[rank]
        PopFitness = PopFitness[rank]

        # 更新最优
        BestPop = Pop[0].copy()  # 全局最优
        BestFitness = PopFitness[0]  # 全局最优值
        HistFitness[It+1] = BestFitness

    #     # 展示迭代信息
    #     print("Iteration {}: Best func = {}".format(It, BestFitness))
    # 运行结果
    end = time.time()  # 运行结束时刻

    # print("最优解：{}".format(BestPop))
    # print("最优值：{}".format(BestFitness))
    # print('运行时间：{}s'.format(end - start))
    # print(BestPop)
    if His:
        return BestFitness, HistFitness
    else:
        return BestFitness


def CMAES(func, nVar,VarMin,VarMax, nPop, MaxIt,S=0,His=0):
    lmd=nPop
    mu = lmd // 2  # 选择的后代数量
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)  # 后代的组合权重
    muw = 1 / np.sum(weights ** 2)  # 后代方差有效数量
    cc = (4 + muw / nVar) / (4 + nVar + 2 * muw / nVar)  # 演化路径的学习率
    cs = (muw + 2) / (nVar + muw + 5)  # 方差更新的学习率
    c1 = 2 / ((nVar + 1.3) ** 2 + muw)  # rank-1-update的学习率
    cmu = np.min([1 - c1, muw / nVar ** 2])  # rank-mu-update的学习率
    ds = 1 + cs + 2 * np.max(np.sqrt((muw - 1) / (nVar + 1)) - 1, 0)  # 方差更新的阻尼因子

    # 分布参数初始化
    xmean = np.random.random(nVar)  # 初始化均值
    sigma = (VarMax - VarMin) / 2
    pc = np.zeros(nVar)  # C的进化路径
    ps = np.zeros(nVar)  # sigma的进化路径
    B = np.eye(nVar)  # 协方差矩阵的特征矩阵
    D = np.ones(nVar)  # 协方差矩阵的特征值
    C = B @ np.diag(D ** 2) @ B.T  # 协方差矩阵
    invsqrtC = B @ np.diag(D ** (-1)) @ B.T  # C^-1/2
    chiN = np.sqrt(nVar)  # E||N(0,I)||
    idx = np.arange(lmd)  # 用于排序


    if isinstance(S,np.ndarray):# 初始化种群
        ind=func(S).argsort()[:nPop]
        Pop=S[ind].copy()
    else:
        Pop = np.random.uniform(VarMin, VarMax, [lmd, nVar])
    PopFitness = func(Pop)
    HistFitness = np.zeros(MaxIt+1)  # 记录最优
    HistFitness[0] = min(PopFitness)

    # 开始搜索
    start = time.time()
    for It in range(MaxIt):
        Pop = xmean + sigma * np.random.randn(lmd, nVar) @ np.diag(D) @ B
        Pop = np.clip(Pop, VarMin, VarMax)  # 修正边界
        PopFitness = func(Pop)

        # 选择较优的mu个个体,并更新xmean
        good_idx = idx[PopFitness.argsort()][:mu]
        xold = xmean
        xmean = weights @ Pop[good_idx]

        # 更新进化路线
        pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * muw) * (xmean - xold) / sigma
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * muw) * (xmean - xold) @ invsqrtC / sigma

        # 更新协方差矩阵
        y = (Pop[good_idx] - xold) / sigma
        sigmay = 0  # 协方差加权求和
        for i in range(mu):
            sigmay = sigmay + weights[i] * y[i].reshape([nVar, 1]) @ y[i].reshape([1, nVar])
        C = (1 - c1 - cmu) * C + c1 * pc.reshape([nVar, 1]) @ pc.reshape([1, nVar]) + cmu * sigmay

        # 更新sigma
        sigma = sigma * np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))
        #         # print(sigma)
        # 更新B和D
        #         # print(C, end='\n\n\n\n')
        C = np.triu(C) + np.triu(C, 1).T
        D, B = np.linalg.eig(C + 0.001)
        D = np.real(D)
        B = np.real(B)
        D = np.sqrt(D)
        invsqrtC = B @ np.diag(D ** (-1)) @ B.T

        # 记录
        if min(PopFitness)<HistFitness[It]:
            HistFitness[It+1] = min(PopFitness)
        else:
            HistFitness[It + 1] =HistFitness[It]

        # 展示迭代信息
    #         # print("Iteration {}: Best func = {}".format(It, HistFitness[It]))

    # 运行结果
    end = time.time()  # 运行结束时刻
    BestPop = Pop[np.argmin(PopFitness)].copy()  # 全局最优解
    BestFitness = min(HistFitness)  # 全局最优值
    # print("最优解：{}".format(BestPop))
    # print("最优值：{}".format(BestFitness))
    # print('运行时间：{}s'.format(end - start))

    if His:
        return BestFitness, HistFitness
    else:
        return BestFitness


def ES(func, nVar,VarMin,VarMax, nPop, MaxIt,S=0,His=0):

    # 参数配置
    nKid = 30  # 用于迭代的孩子

    # 初始化
    if isinstance(S,np.ndarray):# 初始化种群
        ind=func(S).argsort()[:nPop]
        mu=S[ind].copy()
    else:
        mu = (VarMax + VarMin) / 2  # 均值
    sigma = (VarMax - VarMin) / 2  # 标准差
    HistFitness = np.zeros(MaxIt+1)  # 记录最优
    HistFitness[0]=np.min(mu)
    # 开始进化
    start = time.time()
    for It in range(MaxIt):
        # 通过均值标准差来生成子代
        Pop = mu + sigma * np.random.randn(nPop, nVar)
        Pop = np.clip(Pop, VarMin, VarMax)  # 修正边界
        PopFitness = func(Pop)
        idx = np.arange(Pop.shape[0])
        # 选择较优的nKid个个体
        good_idx = idx[PopFitness.argsort()][:nKid]
        Kid = Pop[good_idx].copy()

        # 用nKid个个体来更新均值与标准差
        mu = np.mean(Kid, axis=0)
        sigma = np.std(Kid, axis=0)
        HistFitness[It+1] = min(PopFitness)
    BestPop = Pop[np.argmin(PopFitness)].copy()  # 全局最优解
    BestFitness = min(PopFitness)  # 全局最优值

    # print(BestPop)
    if His:
        return BestFitness, HistFitness
    else:
        return BestFitness



def CSO(func, nVar,VarMin,VarMax, nPop, MaxIt,S=0,His=0):

    # 参数配置
    VMax = 0.2 * (VarMax - VarMin)  # 最大速度
    phi = 0.01  # 社会因素

    # 种群初始化
    if isinstance(S,np.ndarray):
        ind=func(S).argsort()[:nPop]
        Pop=S[ind].copy()
    else:
        Pop = np.random.uniform(VarMin, VarMax, [nPop, nVar])  # 种群初始化
    V = np.random.uniform(-VMax, VMax, [nPop, nVar])  # 速度初始化
    PopFitness = func(Pop)
    BestPop = Pop[np.argmin(PopFitness)].copy()  # 全局最优
    BestFitness = min(PopFitness)  # 全局最优值
    HistFitness = np.zeros(2*MaxIt + 1)  # 记录最优
    HistFitness[0] = BestFitness

    # 搜索
    start = time.time()
    for It in range(2 * MaxIt):
        # 两两相互竞争选出胜者与败者
        index = np.random.choice(range(nPop), nPop, replace=False)
        competitor1 = index[:nPop // 2]
        competitor2 = index[nPop // 2:]

        winner = list(competitor1[PopFitness[competitor1] <= PopFitness[competitor2]]) + list(
            competitor2[PopFitness[competitor2] < PopFitness[competitor1]])
        loser = list(competitor1[PopFitness[competitor1] > PopFitness[competitor2]]) + list(
            competitor2[PopFitness[competitor2] >= PopFitness[competitor1]])

        # 更新败者的位置
        R1 = np.random.random([nPop // 2, nVar])
        R2 = np.random.random([nPop // 2, nVar])
        R3 = np.random.random([nPop // 2, nVar])
        x_bar = np.mean(Pop, axis=0)
        V[loser] = R1 * V[loser] + R2 * (Pop[winner] - Pop[loser]) + phi * R3 * (x_bar - Pop[loser])
        Pop[loser] = Pop[loser] + V[loser]

        Pop[Pop > VarMax] = VarMax  # 修正位置边缘
        Pop[Pop < VarMin] = VarMin

        # 计算适应度
        PopFitness[loser] = func(Pop[loser])

        if min(PopFitness) < BestFitness:
            BestPop = Pop[np.argmin(PopFitness)].copy()  # 全局最优解
            BestFitness = min(PopFitness)  # 全局最优值

        HistFitness[It+1]=BestFitness
    #     # 展示迭代信息
    #     print("Iteration {}: Best func = {}".format(It, BestFitness))
    #
    # # 运行结果
    # end = time.time()  # 运行结束时刻
    #
    # print("最优解：{}".format(BestPop))
    # print("最优值：{}".format(BestFitness))
    # print('运行时间：{}s'.format(end - start))
    if His:
        return BestFitness, HistFitness
    else:
        return BestFitness


def DE(func, nVar,VarMin,VarMax, nPop, MaxIt,S=0,His=0):

    # 参数配置
    F = 0.5  # 缩放因子

    # 种群初始化
    if isinstance(S,np.ndarray):# 初始化种群
        ind=func(S).argsort()[:nPop]
        Pop=S[ind].copy()
    else:
        Pop = np.random.uniform(VarMin, VarMax, [nPop, nVar])  # 种群初始化
    PopFitness = func(Pop)
    BestPop = Pop[np.argmin(PopFitness)].copy()  # 全局最优
    BestFitness = min(PopFitness)  # 全局最优值
    HistFitness = np.zeros(MaxIt+1)  # 记录最优
    HistFitness[0]=BestFitness

    # 搜索
    start = time.time()
    for It in range(MaxIt):
        # 变异
        H = Pop[np.random.choice(range(nPop), nPop)] + F * (
                Pop[np.random.choice(range(nPop), nPop)] - Pop[np.random.choice(range(nPop), nPop)])
        H = np.clip(H, VarMin, VarMax)  # 修正边界

        # 交叉
        V = np.zeros([nPop, nVar])
        crossover = np.random.random([nPop, nVar])
        V[crossover > 0.5] = Pop[crossover > 0.5].copy()
        V[crossover <= 0.5] = H[crossover <= 0.5].copy()

        # 选择
        VFitness = func(V)
        Pop[VFitness < PopFitness] = V[VFitness < PopFitness].copy()

        # 计算适应度
        PopFitness[VFitness < PopFitness] = VFitness[VFitness < PopFitness]

        if min(PopFitness) < BestFitness:
            BestPop = Pop[np.argmin(PopFitness)].copy()  # 全局最优解
            BestFitness = min(PopFitness)  # 全局最优值
        HistFitness[It+1] = BestFitness

    # # 运行结果
    # end = time.time()  # 运行结束时刻
    #
    # print("最优解：{}".format(BestPop))
    # print("最优值：{}".format(BestFitness))
    # print('运行时间：{}s'.format(end - start))
    if His:
        return BestFitness, HistFitness
    else:
        return BestFitness


def FEP(func, nVar,VarMin,VarMax, nPop, MaxIt,S=0,His=0):
    # 初始化
    if isinstance(S,np.ndarray):# 初始化种群
        ind=func(S).argsort()[:nPop]
        POP=S[ind].copy()
        Pop = dict(DNA=POP,  # DNA初始化
                   Mut=np.random.rand(nPop, nVar))  # 记录每一个DNA的变异率
    else:
        Pop = dict(DNA=np.random.uniform(VarMin, VarMax, [nPop, nVar]),  # DNA初始化
               Mut=np.random.rand(nPop, nVar))  # 记录每一个DNA的变异率
    PopFitness = func(Pop['DNA'])
    HistFitness = np.zeros(MaxIt+1)  # 记录最优
    HistFitness[0] = np.min(PopFitness)

    # 开始进化
    start = time.time()
    for It in range(MaxIt):
        # # 通过变异产生子代
        kids = {'DNA': Pop['DNA'] + Pop['Mut'] * np.random.standard_cauchy([nPop, nVar]),
                'Mut': Pop['Mut'] * np.exp(np.sqrt(2 * np.sqrt(nVar)) ** -1 * np.random.randn(nPop, 1) + np.sqrt(
                    2 * nVar) ** -1 * np.random.randn(nPop, nVar))}
        kids['DNA'] = np.clip(kids['DNA'], VarMin, VarMax)

        # 通过排序法选出下一代的父代
        Pop['DNA'] = np.vstack((Pop['DNA'], kids['DNA']))  # 整合本代个体
        Pop['Mut'] = np.vstack((Pop['Mut'], kids['Mut']))
        PopFitness = np.vstack((PopFitness, func(kids['DNA']))).reshape([-1])
        idx = np.arange(Pop['DNA'].shape[0])
        good_idx = idx[PopFitness.argsort()][:nPop]  # 选择较优的nPop个个体
        Pop['DNA'] = Pop['DNA'][good_idx].copy()
        Pop['Mut'] = Pop['Mut'][good_idx].copy()
        PopFitness = PopFitness[good_idx].copy()
        HistFitness[It+1] = min(PopFitness)

    # 运行结果
    end = time.time()  # 运行结束时刻

    BestPop = Pop['DNA'][np.argmin(PopFitness)].copy()  # 全局最优解
    BestFitness = min(PopFitness)  # 全局最优值
    # print("最优解：{}".format(BestPop))
    # print("最优值：{}".format(BestFitness))
    # print('运行时间：{}s'.format(end - start))
    if His:
        return BestFitness, HistFitness
    else:
        return BestFitness


def GA(func, nVar,VarMin,VarMax, nPop, MaxIt,S=0,His=0): # 实数编码GA

    # 种群初始化
    if isinstance(S,np.ndarray):# 初始化种群
        ind=func(S).argsort()[:nPop]
        Pop=S[ind].copy()
    else:
        Pop = np.random.uniform(VarMin, VarMax, [nPop, nVar])  # 种群初始化
    PopFitness = func(Pop)
    BestPop = Pop[np.argmin(PopFitness)].copy()  # 全局最优
    BestFitness = min(PopFitness)  # 全局最优值
    HistFitness = np.zeros(MaxIt+1)  # 记录最优
    HistFitness[0]=BestFitness
    idx = np.arange(2 * nPop)

    # 遗传过程
    start = time.time()
    for It in range(MaxIt):
        # 选择父代 二元锦标赛
        parent = []
        for n in range(nPop):
            competitor = np.random.randint(0, nPop, 2)
            if PopFitness[competitor[0]] > PopFitness[competitor[1]]:
                parent.append(Pop[competitor[1]].copy())
            else:
                parent.append(Pop[competitor[0]].copy())
        parent1 = np.array(parent[:nPop // 2])
        parent2 = np.array(parent[nPop // 2:])

        # 模拟二进制交叉
        beta = np.zeros(parent1.shape)
        mu = np.random.random(parent1.shape)
        beta[mu <= 0.5] = (2 * mu[mu <= 0.5]) ** (1 / 11)
        beta[mu > 0.5] = (2 - 2 * mu[mu > 0.5]) ** (-1 / 21)
        child = np.zeros([nPop, nVar])
        child[:nPop // 2] = (parent1 + parent2) / 2 + beta * (parent1 - parent2) / 2
        child[nPop // 2:] = (parent1 + parent2) / 2 - beta * (parent1 - parent2) / 2

        # 多项式变异
        site = np.random.random([nPop, nVar]) < 1 / nVar
        mu = np.random.random([nPop, nVar])
        temp = site & (mu <= 0.5)
        child[temp] = child[temp] + (VarMax - VarMin) * ((2 * mu[temp] + (1 - 2 * mu[temp]) * (
                1 - (child[temp] - VarMin) / (VarMax - VarMin)) ** 11) ** (1 / 11) - 1)
        temp = site & (mu > 0.5)
        child[temp] = child[temp] + (VarMax - VarMin) * (1 - (2 * (1 - mu[temp]) + 2 * (mu[temp] - 0.5) * (
                1 - (VarMax - child[temp]) / (VarMax - VarMin)) ** 11) ** (1 / 11))
        child = np.clip(child, VarMin, VarMax)  # 修正边界

        # 选择
        Pop = np.vstack((Pop, child))
        PopFitness = np.vstack((PopFitness, func(child))).reshape([-1])
        rank = idx[PopFitness.argsort()][:nPop]
        Pop = Pop[rank]
        PopFitness = PopFitness[rank]
        # 更新最优
        if min(PopFitness) < BestFitness:
            BestPop = Pop[np.argmin(PopFitness)].copy()  # 全局最优个体
            BestFitness = min(PopFitness)  # 全局最优值
        HistFitness[It+1] = BestFitness
    #
    # # 运行结果
    # end = time.time()  # 运行结束时刻
    #
    # print("最优解：{}".format(BestPop))
    # print("最优值：{}".format(BestFitness))
    # print('运行时间：{}s'.format(end - start))
    # print(BestPop)
    if His:
        return BestFitness, HistFitness
    else:
        return BestFitness


def PSO(func, nVar,VarMin,VarMax, nPop, MaxIt,S=0,His=0):

    # 种群初始化
    if isinstance(S,np.ndarray):# 初始化种群
        ind=func(S).argsort()[:nPop]
        Pop=S[ind].copy()
    else:
        Pop = np.random.uniform(VarMin, VarMax, [nPop, nVar])  # 种群初始化
    V = np.zeros([nPop, nVar])  # 速度初始化
    PopFitness = func(Pop)
    pBest = Pop.copy()  # 个体最优
    fp = PopFitness.copy()
    BestPop = Pop[np.argmin(PopFitness)].copy()  # 全局最优
    BestFitness = min(PopFitness)  # 全局最优值
    HistFitness = np.zeros(MaxIt+1)  # 记录最优
    HistFitness[0]=BestFitness

    # 搜索
    start = time.time()
    for It in range(MaxIt):
        # 更新速度
        r1 = np.random.random([nPop, 1])  # 学习权重的随机因子
        r2 = np.random.random([nPop, 1])
        V = 0.4 * V + r1 * (pBest - Pop) + r2 * (BestPop - Pop)

        # 更新位置
        Pop = Pop + V
        Pop[Pop > VarMax] = VarMax  # 修正位置边缘
        Pop[Pop < VarMin] = VarMin

        # 计算适应度
        PopFitness = func(Pop)
        # 更新个体最优和群体最优
        fp[PopFitness < fp] = PopFitness[PopFitness < fp].copy()
        pBest[PopFitness < fp] = Pop[PopFitness < fp].copy()
        if min(PopFitness) < BestFitness:
            BestPop = Pop[np.argmin(PopFitness)].copy()  # 全局最优解
            BestFitness = min(PopFitness)  # 全局最优值
        HistFitness[It+1] = BestFitness
    # # 运行结果
    # end = time.time()  # 运行结束时刻
    # print("最优解：{}".format(BestPop))
    # print("最优值：{}".format(BestFitness))
    # print('运行时间：{}s'.format(end - start))

    if His:
        return BestFitness, HistFitness
    else:
        return BestFitness


def SA(func, nVar,VarMin,VarMax, nPop, MaxIt,S=0,His=0):

    # 参数配置
    T = 0.1  # 初始温度
    alpha = 0.99  # 降温系数
    L = 10

    # 初始化
    if isinstance(S,np.ndarray):# 初始化种群
        ind=func(S).argsort()[:1]
        x=S[ind].copy()
    else:
        x = np.random.uniform(VarMin, VarMax, [1, nVar])
    fitness = func(x)
    Bestx = x.copy()  # 全局最优
    Bestfitness = fitness[0]  # 全局最优值
    HistT = [T]
    HistFitness = np.zeros(nPop * MaxIt + 1)
    HistFitness[0] = Bestfitness  # 记录最优

    # 搜索
    start = time.time()
    for It in range(nPop * MaxIt):
        xnew = x + T * np.random.uniform(VarMin, VarMax, [1, nVar])
        xnew = np.clip(xnew, VarMin, VarMax)  # 修正边
        fitnessnew = func(xnew)[0]
        if np.random.random() < np.exp((fitness - fitnessnew) / T):
            x = xnew
            fitness = fitnessnew
        Bestx = x
        Bestfitness = fitness
        T = T * alpha
        HistT.append(T)
        HistFitness[It+1]=Bestfitness

    # # 运行结果
    # end = time.time()  # 运行结束时刻
    # # print(T)
    # print("最优解：{}".format(Bestx))
    # print("最优值：{}".format(Bestfitness))
    # print('运行时间：{}s'.format(end - start))
    # print(Bestx)
    if His:
        return Bestfitness, HistFitness
    else:
        return Bestfitness


def RAND(func, nVar,VarMin,VarMax, nPop, MaxIt,S=0,His=0):

    # 种群初始化
    if isinstance(S,np.ndarray):# 初始化种群
        ind=func(S).argsort()[:1]
        x=S[ind].copy()
    else:
        x=np.random.uniform(VarMin, VarMax, [1, nVar])
    fx=func(x)
    Pop = np.random.uniform(VarMin, VarMax, [nPop * MaxIt+1, nVar])  # 种群初始化

    # 随机
    start = time.time()
    PopFitness = func(Pop)
    # BestPop = Pop[np.argmin(PopFitness)].copy()  # 全局最优
    BestFitness = min(PopFitness)  # 全局最优值
    HistFitness = np.zeros(nPop * MaxIt + 1)
    HistFitness[0]=fx
    for i in range(nPop*MaxIt):
        if PopFitness[i]<HistFitness[i]:
            HistFitness[i+1]=PopFitness[i]
        else:
            HistFitness[i + 1] = HistFitness[i]
    # end = time.time()  # 运行结束时刻
    #
    # print("最优解：{}".format(BestPop))
    # print("最优值：{}".format(BestFitness))
    # print('运行时间：{}s'.format(end - start))
    if His:
        return BestFitness, HistFitness
    else:
        return BestFitness
