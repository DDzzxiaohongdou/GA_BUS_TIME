# coding: utf-8
"""
作者：DDZZxiaohongdou	日期：2021/3//5
"""
from __future__ import division
import numpy as np
import random
import math
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import time

class GA(object):
    def __init__(self, maxiter, sizepop, lenchrom, pc, pm, dim, lb, ub, Fobj):
        self.maxiter = maxiter          #最大迭代次数
        self.sizepop = sizepop          #种群数量
        self.lenchrom = lenchrom        #染色体长度
        self.pc = pc                    #交叉概率
        self.pm = pm                    #变异概率
        self.dim = dim                  #变量的维度
        self.lb = lb                    #最小取值
        self.ub = ub                    #最大取值
        self.Fobj = Fobj                #价值函数

    # 初始化种群：返回一个三维数组，第一维是个体，第二维是个体维度，第三维是染色体
    def Initialization(self):
        pop = []
        for i in range(self.sizepop):
            temp1 = []
            for j in range(self.dim):
                temp2 = []
                for k in range(self.lenchrom):
                    temp2.append(random.randint(0, 1))
                temp1.append(temp2)
            pop.append(temp1)
        return pop

    # 将二进制转化为十进制
    def b2d(self, pop_binary):
        pop_decimal = []
        for i in range(len(pop_binary)):
            temp1 = []
            for j in range(self.dim):
                temp2 = 0
                for k in range(self.lenchrom):                     #将每个染色体进行二进制->十进制
                    temp2 += pop_binary[i][j][k] * math.pow(2, k)
                temp2 = temp2 * (self.ub[j] - self.lb[j]) / (math.pow(2, self.lenchrom) - 1) + self.lb[j]  #等比例缩放
                temp1.append(temp2)
            pop_decimal.append(temp1)
        return pop_decimal

    # 轮盘赌模型选择适应值较高的种子
    def Roulette(self, fitness, pop):
        # 适应值按照大小排序
        sorted_index = np.argsort(fitness)
        sorted_fitness, sorted_pop = [], []
        for index in sorted_index:
            sorted_fitness.append(fitness[index])
            sorted_pop.append(pop[index])

        # 生成适应值累加序列
        fitness_sum = sum(sorted_fitness)
        accumulation = [None for col in range(len(sorted_fitness))]                     #累加概率序列
        accumulation[0] = sorted_fitness[0] / fitness_sum
        for i in range(1, len(sorted_fitness)):
            accumulation[i] = accumulation[i - 1] + sorted_fitness[i] / fitness_sum

        # 轮盘赌
        roulette_index = []
        for j in range(len(sorted_fitness)):         #对排过序的适应值进行排序
            p = random.random()                       #生成一个0-1的随机值
            for k in range(len(accumulation)):       #循环，直到找到比这个概率大的区间，选择该区间对应的索引
                if accumulation[k] >= p:
                    roulette_index.append(k)
                    break
        temp1, temp2 = [], []
        for index in roulette_index:              #这里面也许有重复的
            temp1.append(sorted_fitness[index])
            temp2.append(sorted_pop[index])
        newpop = [[x, y] for x, y in zip(temp1, temp2)]
        newpop.sort()
        newpop_fitness = [newpop[i][0] for i in range(len(sorted_fitness))]
        newpop_pop = [newpop[i][1] for i in range(len(sorted_fitness))]
        return newpop_fitness, newpop_pop

    # 交叉繁殖：针对每一个个体，随机选取另一个种子与之交叉。
    # 随机取种子基因上的两个位置点，然后互换两点之间的部分
    def Crossover(self, pop):
        newpop = []
        for i in range(len(pop)):
            if random.random() < self.pc:
                # 选择另一个种子
                j = i
                while j == i:
                    j = random.randint(0, len(pop) - 1)
                cpoint1 = random.randint(1, self.lenchrom - 1)
                cpoint2 = cpoint1
                while cpoint2 == cpoint1:
                    cpoint2 = random.randint(1, self.lenchrom - 1)
                cpoint1, cpoint2 = min(cpoint1, cpoint2), max(cpoint1, cpoint2)
                newpop1, newpop2 = [], []
                for k in range(self.dim):
                    temp1, temp2 = [], []
                    temp1.extend(pop[i][k][0:cpoint1])
                    temp1.extend(pop[j][k][cpoint1:cpoint2])
                    temp1.extend(pop[i][k][cpoint2:])
                    temp2.extend(pop[j][k][0:cpoint1])
                    temp2.extend(pop[i][k][cpoint1:cpoint2])
                    temp2.extend(pop[j][k][cpoint2:])
                    newpop1.append(temp1)
                    newpop2.append(temp2)
                newpop.extend([newpop1, newpop2])
        return newpop

    # 变异：针对每一个种子的每一个维度，进行概率变异，变异基因为一位
    def Mutation(self, pop):
        newpop = copy.deepcopy(pop)
        for i in range(len(pop)):
            for j in range(self.dim):
                if random.random() < self.pm + i*0.0001:             #变化
                    mpoint = random.randint(0, self.lenchrom - 1)
                    newpop[i][j][mpoint] = 1 - newpop[i][j][mpoint]
        return newpop

    # 绘制迭代-误差图
    def Ploterro(self, Convergence_curve):
        mpl.rcParams['font.sans-serif'] = ['Courier New']
        mpl.rcParams['axes.unicode_minus'] = False
        fig = plt.figure(figsize=(10, 6))
        x = [i for i in range(len(Convergence_curve))]
        plt.plot(x, Convergence_curve, 'r-', linewidth=1.5, markersize=5)
        plt.xlabel(u'Iter', fontsize=18)
        plt.ylabel(u'Best score', fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim(0, )
        plt.grid(True)
        plt.show()

    def Run(self):
        pop = self.Initialization()
        errolist = []
        for Current_iter in range(self.maxiter):
            print("Iter = " + str(Current_iter))
            pop1 = self.Crossover(pop)   #交叉               #M*D*L
            pop2 = self.Mutation(pop1)   #变异               #M*D*L
            pop3 = self.b2d(pop2)        #二进制转十进制     #M*D
            fitness = []
            for j in range(len(pop2)):
                fitness.append(self.Fobj(pop3[j]))                         #计算出每个个体的适应值
            sorted_fitness, sorted_pop = self.Roulette(fitness, pop2)      #轮盘赌进行选择
            best_fitness = sorted_fitness[-1]                              #适应度最高的个体
            best_pos = self.b2d([sorted_pop[-1]])[0]                       #适应度最高的个体对应的价值函数的值
            pop = sorted_pop[-1:-(self.sizepop + 1):-1]
            errolist.append(1 / best_fitness)
            print(" best_fitness = " + str(best_fitness))
            print(" cost         = " + str(1/best_fitness))
            if 1 / best_fitness < 0.0001:                                  #提前迭代停止算法
                print("Best_score = " + str(round(1 / best_fitness, 4)))
                print("Best_pos = " + str([round(a, 4) for a in best_pos]))
                break
        return best_fitness, best_pos, errolist


if __name__ == "__main__":

    #计算乘客等待时间，代价函数的一部分
    def times(ptime, btime):
        tsum = 0
        for x in ptime:
            for y in btime:
                if x < y:
                    tsum += y - x
                    break
        return tsum

    # 将发车时间转化为公交到达各站时间btime
    def transfer(bus_time):
        #到达各站所需的时间
        drive = [0, 180, 360, 720, 960, 1260, 1500, 1680, 1860, 2040, 2160, 2280, 2520, 2820, 2940, 3120, 3240, 3360]
        #二维列表，一组公交车到各站的时间
        bus_st3 = [[[]]]
        for i in range(0,42,7):
            bus_st2 = [[]]
            for i in range(18):
                bus_st2.append(bus_time[i:i+7] + drive[i])   #每一批
            bus_st3.append(bus_st2[1:])
        return bus_st3[1:]      #(18*7), 每一行是七列班车在某站的发车时刻  -> 6*18*7

    # 将换成乘客数量转化为乘客到达各站时间ptime 二维数组
    def numtotime(num):  # (18*4)  #18站，4个时间区间
        t = [[]]
        for i in range(len(num)):
            temp = list(np.rint(np.linspace(0, 900, num=int(num[i][0]), endpoint=False)))
            temp.extend(list(np.rint(np.linspace(900, 1800, num=int(num[i][1]), endpoint=False))))
            temp.extend(list(np.rint(np.linspace(1800, 2700, num=int(num[i][2]), endpoint=False))))
            temp.extend(list(np.rint(np.linspace(2700, 3600, num=int(num[i][3]), endpoint=False))))
            t.append(temp)
        return t[1:]


    def Fobj(factor):

        # 普通乘客, 六条线路
        pasnor = [[158, 210, 111, 191],
                  [186, 193, 155, 184],
                  [67,  114, 109, 108],
                  [287, 220, 273, 189],
                  [76,  92,  77,  57],
                  [145, 189, 196, 174]]#普通乘客在一个小时的四个时间区间内的人员分布

        ranseed = [10, 14, 15, 13]       #四个随机种子

        # 换乘乘客 六条线路 每条线路换乘站个数还不一样   6*N*4
        pas_tr = [[[0, 2, 3, 1], [2, 2, 2, 2], [25, 20, 48, 31]],
                  [[18, 18, 10, 18], [13, 23, 16, 15]],
                  [[3, 4, 2, 9], [6, 18, 7, 8], [0, 1, 1, 0]],
                  [[1, 3, 2, 1], [17, 14, 35, 29]],
                  [[14, 7, 16, 21], [3, 2, 6, 2]],
                  [[7, 10, 17, 9]]]

        bustime_tr_index = [[2,5,8],
                            [2,3],
                            [13,14,15],
                            [7,10],
                            [3,4],
                            [2]]

        cost = 0


        for k in range(6):                  #六条公交线路，循环六次

            pas = [[]]                       #四个时间区段中，按照随机种子分布在各站点的人数
            for i in range(4):
                np.random.seed(ranseed[i])
                a = np.random.poisson(lam=10, size=18)             # 泊松分布
                b = np.rint(a / (a.sum(axis=0)) * pasnor[k][i])
                pas.append(b)

            passe = np.array(pas[1:])  # 第一站不考虑等待时间  (4*18)

            pasT = passe.T  # 普通乘客各站人数  (18*4)
            pas_n_t = numtotime(pasT)  # 服从泊松分布的4个时间区间的18站普通乘客分布(18*4)

            pas_tr_t = numtotime(pas_tr[k])

            # 每组个体即为公交发车时间
            bustime = transfer(np.array([factor[0],factor[1],factor[2],factor[3],factor[4],factor[5],factor[6],
                                         factor[7], factor[8], factor[9], factor[10], factor[11], factor[12], factor[13],
                                         factor[14], factor[15], factor[16], factor[17], factor[18], factor[19], factor[20],
                                         factor[21], factor[22], factor[23], factor[24], factor[25], factor[26], factor[27],
                                         factor[28], factor[29], factor[30], factor[31], factor[32], factor[33], factor[34],
                                         factor[35], factor[36], factor[37], factor[38], factor[39], factor[40], factor[41]]))

            bustime_tr = []
            #bustime[k]的维度18*7
            for index_ in bustime_tr_index[k]:
                bustime_tr.append(bustime[k][index_])

            time1 = 0
            time2 = 0

            for i in range(len(pas_n_t)):
                # print(np.array(bustime[k][i]).shape) #7
                # print(np.array(pas_n_t[i]).shape)    #N
                time1 += times(pas_n_t[i], bustime[k][i])

            for i in range(len(pas_tr_t)):

                time2 += times(pas_tr_t[i], bustime_tr[i])

            cost += (time1 + 10 * time2)/600 + 3150

        return 1/cost





    starttime = time.time()

    a = GA(1000, 50, 10, 0.8, 0.01, 42,
           [0, 300, 600, 1200, 1800, 2100, 3000,
            0, 300, 600, 1200, 1800, 2100, 3000,
            0, 300, 600, 1200, 1800, 2100, 3000,
            0, 300, 600, 1200, 1800, 2100, 3000,
            0, 300, 600, 1200, 1800, 2100, 3000,
            0, 300, 600, 1200, 1800, 2100, 3000],
           [500, 800, 1200, 1800, 2100, 3000, 3600,
            500, 800, 1200, 1800, 2100, 3000, 3600,
            500, 800, 1200, 1800, 2100, 3000, 3600,
            500, 800, 1200, 1800, 2100, 3000, 3600,
            500, 800, 1200, 1800, 2100, 3000, 3600,
            500, 800, 1200, 1800, 2100, 3000, 3600],
           Fobj)

    Best_score, Best_pos, errorlist = a.Run()

    print(Best_score)
    print(Best_pos)
    print(errorlist)

    endtime = time.time()

    print("Runtime = " + str(endtime - starttime))
    a.Ploterro(errorlist)
