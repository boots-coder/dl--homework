import numpy as np
"""
To enhance my understanding of this assignment,
this code specifically points out the critical part: how to extract the X1000 samples.
"""

# first we build some test data
data = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0,])
labels = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0,])

# then init blank list
new_data = []
new_labels = []
'''
# 定义一个包含单个元素的元组
a = ([1, 2, 3],)

# 打印元组的第一个元素
print(a[0])  # 输出：[1, 2, 3]

# 打印整个元组
print(a)  # 输出：([1, 2, 3],)
'''
for digit in range(5):
    idx = np.where(labels == digit)[0][:1]
    whole = np.where(labels == digit)
    # we can check idx for enhence understanding of this line or you can debug for this line
    print("idx:" + str(idx))
    new_data.extend(data[idx])
    new_labels.extend(labels[idx])
# In summary, there is nothing surprising about this usage;
# it is important to note the use of tuples in Python syntax, and that `numpy.where(condition)` will return a tuple.
new_data = np.array(new_data)
new_labels = np.array(new_labels)
# [0 1 2 3 4]
# [0 1 2 3 4]
print(new_data)
print(new_labels)
