import torch

class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim) # 缩放参数，初始为 1
        self.beta = torch.zeros(dim) # 平移参数，初始为 0
        self.running_mean = torch.zeros(dim) # 运行时的均值
        self.running_var = torch.ones(dim) # 运行时的方差
    def __call__(self, x):
        if self.training:
            xmean = x.mean(0,keepdim=True) # 计算当前批次的均值
            xvar = x.var(0,keepdim=True) # 计算当前批次的方差
        else:
            mean = self.running_mean
            var = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # 标准化
        self.out = self.gamma * xhat + self.beta # 缩放和平移
        return self.out
    def parameters(self):
        return [self.gamma, self.beta]

# 测试流程，演示内部变量变化
torch.manual_seed(0)
module = BatchNorm1d(3)
print('module.training (初始):', module.training)
print('running_mean (初始):', module.running_mean)
print('running_var  (初始):', module.running_var)

x = torch.randn(4,3)
print('\n输入 x:')
print(x)

# 第一次调用（训练模式）
out = module(x)
print('\n--- 第一次调用（training=True） ---')
print('batch mean x.mean(0,keepdim=True):')
print(x.mean(0,keepdim=True))
print('batch var  x.var(0,keepdim=True):')
print(x.var(0,keepdim=True))
print('\ngamma:')
print(module.gamma)
print('beta:')
print(module.beta)
print('\nout (module.out):')
print(module.out)
print('\nrunning_mean (调用后未更新):', module.running_mean)
print('running_var  (调用后未更新):', module.running_var)

# 切换到推理模式，演示代码中 else 分支变量名问题
module.training = False
print('\nmodule.training 设置为 False，准备再次调用：')
try:
    out2 = module(x)
    print('第二次调用输出:', out2)
except Exception as e:
    print('第二次调用时发生错误（预期，代码里 else 分支使用了不同的变量名）:')
    print(e)
