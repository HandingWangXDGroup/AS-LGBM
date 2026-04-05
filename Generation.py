from NODE import NODE
import numpy as np
import copy

#           Meaning
# 0-1       variable                    x[round(i*D)]
# 1-2       constent                    0-10
# 10        Negative                    -x
# 11        Reciprocal                  1/sqrt(1+x.^2)
# 12        Multiplying by 10           10*x
# 13        Square                      x^2
# 14        Square root                 sqrt(abs(x))
# 15        Absolute value              abs(x)
# 16        Rounded value               round(x)
# 17        Sine                        sin(2*pi*x)
# 18        Cosine                      cos(2*pi*x)
# 19        Logarithm                   log(1+abs(x))
# 20        Exponent                    exp(x)
# 21        Addition                    x+y
# 22        Subtraction                 x-y
# 23        Multiplication              x*y
# 24        Analytic Quotient           x/sqrt(1+y^2)

# 定义选择操作符的概率
mOperater = list(range(10, 25))
# -1、1/、10*、^2、sqrt、abs、round、sin、cos、log、exp、+、-、*、/
pOperater = np.array([ 2, 2, 2, 4, 4, 2, 2, 3, 3, 3, 3, 40, 40, 20, 20])
# pOperater = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
pOperater = pOperater / pOperater.sum()


# 生成树
def generate_tree(minlen, maxlen, GorF=1):
    # 初始化树
    tree = NODE(1)
    # 随机树深度
    depth = np.random.randint(minlen, maxlen + 1, 1)[0]

    # 随机生成树,通过Ramped half-and-half方法
    # grow
    if GorF == 0:
        Grow_tree(tree, depth)
    else:
        Full_tree(tree, depth)
    return tree


def Grow_tree(tree, depth):
    while tree.depth() < depth:
        p = tree
        # 随机选择一个叶子节点
        while p.type() > 0:
            if p.type() == 1 or np.random.random() < 0.5:
                p = p.left
            else:
                p = p.right

        # 随机选择一个操作符
        operater = np.random.choice(mOperater, 1, False, pOperater)[0]
        if operater < 20:  # 二元运算符
            if np.random.randint(0, 10, 1)[0] != 1:
                operand = np.random.random()  # 决策变量
            else:
                operand = 1 + np.random.random()  # 常数
            if np.random.random() < 0.5:
                p.left = NODE(p.value)
                p.right = NODE(operand)
            else:
                p.left = NODE(operand)
                p.right = NODE(p.value)
        else:
            p.left = NODE(p.value)
        p.value = operater


def Full_tree(tree, depth):
    if depth <= 1:
        if np.random.randint(0, 10, 1)[0] != 1:
            operand = np.random.random()  # 决策变量
        else:
            operand = 1 + np.random.random()  # 常数
        tree.value = operand
    else:
        operater = np.random.choice(mOperater, 1, False, pOperater)[0]
        tree.value = operater
        if operater <= 20:  # 一元运算符
            tree.left = NODE(1)
            Full_tree(tree.left, depth - 1)
        else: # 二元运算符
            tree.left = NODE(1)
            tree.right = NODE(1)
            Full_tree(tree.left, depth - 1)
            Full_tree(tree.right, depth - 1)



# 树转化为逆波兰式
def tree2expr(tree):
    if tree.type() == 0:
        return [tree.value]
    elif tree.type() == 1:
        return tree2expr(tree.left) + [tree.value]
    else:
        return tree2expr(tree.left) + tree2expr(tree.right) + [tree.value]


def expr2func(expr, D=10, Popcal=False):
    # Popcal表示是否用于numpy进行整个种群的运算，D表示维度

    func = []
    for i in expr:
        if i < 1:  # Real number in 1-10
            if Popcal:
                func += [f"x[:,{int(i * D)}]"]
            else:
                func += [f"x[{int(i * D)}]"]
        elif i < 2:  # const
            func += [f"{10 * (i - 1)}"]
        elif i == 21:  # Addition
            func = func[:-2] + ['({}+{})'.format(func[-2], func[-1])]
        elif i == 22:  # Subtraction
            func = func[:-2] + ['({}-{})'.format(func[-2], func[-1])]
        elif i == 23:  # Multiplication
            func = func[:-2] + ['({}*{})'.format(func[-2], func[-1])]
        elif i == 24:  # AQ
            func = func[:-2] + ['({}/np.sqrt(1+{}**2))'.format(func[-2], func[-1])]
        elif i == 10:  # Negative
            func = func[:-1] + ["(-1*{})".format(func[-1])]
        elif i == 11:  # Reciprocal
            func = func[:-1] + ["(1/np.sqrt(1+{}**2))".format(func[-1])]
        elif i == 12:  # Multiplying by 10
            func = func[:-1] + ["(10*{})".format(func[-1])]
        elif i == 13:  # Square
            func = func[:-1] + ["({}**2)".format(func[-1])]
        elif i == 14:  # Square root
            func = func[:-1] + ["(abs({})**0.5)".format(func[-1])]
        elif i == 15:  # Absolute value
            func = func[:-1] + ["(abs({}))".format(func[-1])]
        elif i == 16:  # Rounded value
            func = func[:-1] + ["(np.round({}))".format(func[-1])]
        elif i == 17:  # Sine
            func = func[:-1] + ["(np.sin(2*np.pi*{}))".format(func[-1])]
        elif i == 18:  # Cosine
            func = func[:-1] + ["(np.cos(2*np.pi*{}))".format(func[-1])]
        elif i == 19:  # Logarithm
            func = func[:-1] + ["(np.log(1+abs({})))".format(func[-1])]
        elif i == 20:  # Exponent
            func = func[:-1] + ["(np.exp({}))".format(func[-1])]
    return func[0]


# def expr2tree(expr):
#     tree = []
#     for i in expr:
#         if i <= 7:
#             tree.append(NODE(i))
#         elif i <= 14:
#             tree = tree[:-2] + [NODE(i, tree[-2], tree[-1])]
#         else:
#             tree = tree[:-1] + [NODE(i, tree[-1])]
#     return tree[0]


if __name__ == '__main__':
    def gt(D):
        tree = NODE(1)
        if D <= 1:
            if np.random.randint(0, 10, 1)[0] != 1:
                operand = np.random.random()  # 决策变量
            else:
                operand = 1 + np.random.random()  # 常数
            tree.value = operand
        else:
            operater = np.random.choice(mOperater, 1, False, pOperater)[0]
            tree.value = operater
            if operater <= 20:  # 一元运算符
                tree.left = gt(D - 1)
            else:
                tree.left = gt(D - 1)
                tree.right = gt(D - 1)

        return tree
    tree = generate_tree(4, 8, 1)
    # tree = gt(3)
    expr = tree2expr(tree)
    expr=[0.655880000000000,0.637110000000000,22,0.752470000000000,19,23,0.633590000000000,0.355490000000000,23,0.512180000000000,15,21,23,0.732210000000000,1.77016000000000,21,0.168410000000000,0.576580000000000,21,21,0.964050000000000,0.0625900000000000,23,0.688380000000000,0.296740000000000,24,22,22,21,0.877620000000000,13,0.384100000000000,0.468210000000000,23,22,1.41647000000000,16,0.205830000000000,13,22,23,14,23,14,0.295640000000000,0.0747900000000000,21,1.18338000000000,0.450960000000000,22,23,1.82419000000000,0.914430000000000,24,0.171340000000000,0.727890000000000,22,24,22,0.402720000000000,16,0.242720000000000,0.0965700000000000,21,23,0.662880000000000,1.04062000000000,24,0.352790000000000,0.926340000000000,23,21,22,22,0.362710000000000,0.520180000000000,24,0.200910000000000,0.508390000000000,21,21,0.111310000000000,14,0.780370000000000,11,22,24,0.129100000000000,17,0.576200000000000,0.970290000000000,23,21,0.243030000000000,0.539670000000000,22,0.378020000000000,1.22102000000000,23,24,24,22,23,1.14235000000000,10,0.678130000000000,0.975560000000000,21,21,0.756830000000000,0.252280000000000,21,0.865360000000000,0.0753600000000000,23,24,22,0.0778200000000000,0.126750000000000,21,0.0808500000000000,0.209660000000000,21,23,0.212480000000000,0.809240000000000,21,0.526360000000000,0.569670000000000,23,22,21,21,0.302950000000000,0.946370000000000,22,0.836040000000000,0.305930000000000,21,21,0.581780000000000,0.528680000000000,21,0.156660000000000,0.0128500000000000,24,22,22,1.58504000000000,0.794460000000000,22,0.868730000000000,0.227120000000000,22,22,0.265300000000000,0.0350900000000000,22,0.534530000000000,19,23,23,21,23,22,23]
    
    func = expr2func(expr, 2,1)
    print(expr)
    print(func)
    print(tree.depth())
    print((np.array(expr)<2).sum())
    X=np.array([[1,2],[3,4],[5,6]])
    f = eval("lambda x: " + func)
    print(f(X))
    # test
    # record = []
    # for i in range(10000):
    #     tree = generate_tree(4, 4, 1)
    #     expr = tree2expr(tree)
    #     a = np.array(expr)
    #     # print((a < 2).sum())
    #     if i % 100 == 0:
    #         print(i)
    #     record.append((a < 2).sum())
    # print(np.mean(record))
    # print(np.std(record))

    pass

# 双目运算符及单目运算符比例 与 叶子节点个数之间的关系
# 深度为8、full
# 双目与单目比例   mean    std
#   0：1         1       0
#   1：1         17      9.45
#   2：1         36      15.8
#   3：1         50      18.8
#   4：1         61      20.3
#   5：1         69      20.8
#   6：1         75      20.9
#   7：1         81      21
#   8：1         86      20.6
#   9：1         89      20.3
#   1：0         128     0

