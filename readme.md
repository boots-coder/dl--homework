### 深度学习作业

本仓库包含深度学习课程的练习代码和十次作业的练习。

目录

- [安装要求](#安装要求)
- [作业列表](#作业列表)
- [环境设置](#环境设置)
- [TensorFlow 相关包](#tensorflow-相关包)

### 安装要求

在开始之前，请确保您的系统中安装了以下软件：

- [Python 3.11](https://www.python.org/downloads/release/python-3110/)
- [Anaconda](https://www.anaconda.com/products/distribution)

### 作业列表

1. 作业 1：基础神经网络应用-minist数字预测
2. 作业 2：编写特定的cnn网络结构

### 环境设置

请按照以下步骤设置您的开发环境：

1. 创建并激活 Conda 环境：

   ```bash
   conda create --name dl_course python=3.11
   conda activate dl_course
   ```

2. 安装 TensorFlow 及相关包：

   ```bash
   pip install tensorflow==2.16.1
   pip install tensorflow-macos==2.16.1
   pip install tensorflow-metal==1.1.0
   ```

3. 安装其他依赖包：

   ```bash
   conda install numpy pandas matplotlib
   conda install -c conda-forge jupyterlab
   ```

### TensorFlow 相关包

以下是与 TensorFlow 相关的包及其版本：

- `tensorflow` 版本：2.16.1
- `tensorflow-io-gcs-filesystem` 版本：0.37.0
- `tensorflow-macos` 版本：2.16.1
- `tensorflow-metal` 版本：1.1.0
- `tensorboard` 版本：2.16.2
- `tensorboard-data-server` 版本：0.7.2



### 开发环境

- macOS 14.4 (23E214)
- pycharm 2024



### 贡献

欢迎提交 pull request 以改进本仓库。如果您有任何建议或问题，请在 Issues 中提出。

