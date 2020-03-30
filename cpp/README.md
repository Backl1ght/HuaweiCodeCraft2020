baseline-GD.cpp: 一开始写的LR+normalize+minibatch

km.cpp: k-means，用牛逼网友开源的NEON代码片段做类型转换

km1.cpp: k-means, 根据数据集特性自己写的读入，会比km.cpp稍慢

mt.cpp: 在km1.cpp的基础上加了多线程优化，0.5s左右

mp.cpp: 在km1.cp的基础上加了多进程优化，0.46s左右