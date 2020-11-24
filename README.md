


### 目标
1. 基于GAN的 音频频域 隐写
2. 数据集：AAC音频的Qmdct系数


### 步骤
1. 经过GAN训练之后，generator G 合成修改概率
2. 概率转化为失真，利用STC做嵌入提取

### 图解
	 ![Fig](https://github.com/LeeeLiu/STC_steg_GAN/blob/master/architecture.png)