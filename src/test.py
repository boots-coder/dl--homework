import gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# 定义和训练时相同的 Q 网络结构
'''运行该代码之前需要安装最新的pygame游戏库'''
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 加载训练好的模型
model = Qnet()
model.load_state_dict(torch.load('cartpole_model_613.pth'))  # 将 xxxx 替换为实际保存的模型文件名
model.eval()  # 将模型设置为评估模式

# 创建CartPole-v1环境
env = gym.make('CartPole-v0', render_mode='human')  # 指定渲染模式为 human

# 初始化变量
state, _ = env.reset()
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # 转换为 tensor 并调整形状
score = 0

# 进行游戏
while True:
    env.render()  # 显示游戏图像
    with torch.no_grad():  # 在评估过程中不需要计算梯度
        action = model(state).argmax().item()  # 预测动作
    next_state, reward, done, truncated, info = env.step(action)  # 执行动作并获取结果
    next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)  # 调整状态形状
    state = next_state  # 更新状态
    print("当前分数为",score)
    score += reward  # 累计分数
    if done:  # 游戏结束
        print("最终得分为: ", score)
        break
env.close()  # 关闭环境