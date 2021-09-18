
## 目标
1. 基于GAN的 信息隐藏

## 方法一：直接，网络合成stego
1. 图解
  - ![HiDDeN](https://papers-1300025586.cos.ap-nanjing.myqcloud.com/GAN/HiDDeN.png)

## 方法二：间接，网络学习失真代价
1. 方法
  - 经过GAN训练之后，generator G 合成修改概率
  - 概率转化为失真，利用STC做嵌入提取

2. 图解
  - ![pro2cost](https://papers-1300025586.cos.ap-nanjing.myqcloud.com/GAN/pro2cost.png)
