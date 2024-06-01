## homework-1



![image-20240601161343604](https://markdown-pictures-jhx.oss-cn-beijing.aliyuncs.com/picgo/image-20240601161343604.png)



Run

```bash
cd /Users/bootscoder/PycharmProjects/dl-homework-practise/homework-1/src #更换为你的src目录
tensorboard --logdir=logs/fit
```

Result:

![image-20240601171228468](https://markdown-pictures-jhx.oss-cn-beijing.aliyuncs.com/picgo/image-20240601171228468.png)

通过对比发现，X1 的训练集较小，出现了过拟合的现象：在训练集的准确率较高，然而在验证集中表现较差

