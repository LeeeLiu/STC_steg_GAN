# pro-cost-STC



### 一、数据集cover、目录说明：
1. 歌曲wav（双声道，44.1Khz, 32位，10s）以128 Kbps码率编码，得到Qmdct（未剪裁）
	（ps：对于单声道，剪裁以后的时长 = 127*1024/16000 = 8.128 s）
2. 目录结构，参见DIRECTORY.txt
	- wav数据集：wav_file
	- 概率图：4000_pro
	- 载体mdct：4000_Qmdct_cover
	- 秘密信息：4000_message
	- 提取出来的信息：4000_extr_msg

### 二、流程
1. 为了得到128*1024尺寸，剪裁wav。剪裁以后时长 = 126*1024/44100/2 = 1.463 s 。（clip.py）
	buf = sig[0: 63*1024, :]    #  因为是双声道

2. 获取128*1024尺寸的Qmdct-cover（batch.py）
	`faac_Qmdct_generate -o  2.m4a  -b 128 wav10s_00002.wav`	# faac在控制台显示64帧（双声道）

3. 将Qmdct-cover送进GAN网络训练，得到修改概率pro.txt

4. 用pro.txt放进‘pro-cost-STC’进行实际STC隐写，调整α，使得嵌入率Kbps标准相同。
	-->得到Qmdct-stego（一维的存为Sqmdct，二维的存为'音频名字.txt'）（lt_tests.m 一维->二维）

5. 将Qmdct-stego和Qmdct-cover放进spec-resnet，检测安全性


### 三、命令示例：
	```
	del de_Sqmdct.txt
	lt_emb_STC   4000_Qmdct_cover\00001.wav.txt   4000_pro\00001.wav.txt    嵌入信息.txt    0.4
	embed_STC    -o  0.7-1-stego.m4a   -b 128   00001.wav
	2-channel-extract_STC    0.7-1-stego.m4a
	lt_extr_STC    de_Sqmdct.txt   n_msg_bits.txt    提取出来的信息.txt
	```

### 四、命令具体含义
1. c编解码器中做嵌入和提取：
	- STC嵌入：embed_STC。利用（matlab生成的exe输出的stego：Sqmdct.txt 放进编码器 替换Qmdct）来嵌入。
	`embed_STC    -o  stegoname.aac     covername.wav`
	- STC提取，得到de_Sqmdct.txt。
	` x-channel-extract_STC    stegoname.aac`

2. STC-mtb 在Qmdct上做嵌入（生成Sqmdct）和提取
	- 嵌入
	`lt_emb_STC    2000_Qmdct_cover\00124.wavQmdct_cover.txt    1500_pro\00124.wavQmdct_cover.txt    秘密信息.txt    0.5`
	- 提取
	`lt_extr_STC     de_Sqmdct.txt    n_msg_bits.txt    提取出来的信息.txt`


### 五、其他
1. Win-操作
	- 生成目录树
	`tree/f>a.txt`
	- 重命名
	`rename    oldname  newname`
	- 移动
	`move	faac\faac.txt（原位置）	.\faad（目标位置）`
	- 复制
	`copy      Cover-1.aac    .\audio`
	- 彻底删除
	`del xxx`

2. linux-操作
	- 打开图片
	`$ eog   a.png  `
	- 查看文件信息（时间等）
	`$ stat    xxx-file`
	- 查看特定程序对应的PID 并杀死进程
	`$ ps aux|grep pycharm`
	`$ kill -9 [PID]       （从左到右第二个号）`
	

