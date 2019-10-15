# pro-cost-STC

### 一、命令
	1. 示例
	```
	del de_Sqmdct.txt
	lt_emb_STC   2000_Qmdct_cover\00124.wavQmdct_cover.txt   1500_pro\00124.wavQmdct_cover.txt    秘密信息.txt    0.1
	embed_STC    -o  00124-stego.aac     00124.wav
	1-channel-extract_STC    00124-stego.aac
	lt_extr_STC    de_Sqmdct.txt   n_msg_bits.txt    提取出来的信息.txt
	```
	2. 解释
	（1）c编解码器中做嵌入和提取：
	a. STC嵌入：embed_STC。利用（matlab生成的exe输出的stego：Sqmdct.txt 放进编码器 替换Qmdct）来嵌入。
	》embed_STC    -o  stegoname.aac     covername.wav
	b. STC提取，得到de_Sqmdct.txt。
	》x-channel-extract_STC    stegoname.aac

	（2）STC-mtb 在Qmdct上做嵌入（生成Sqmdct）和提取
	lt_emb_STC    2000_Qmdct_cover\00124.wavQmdct_cover.txt    1500_pro\00124.wavQmdct_cover.txt    秘密信息.txt    0.5
	lt_extr_STC     de_Sqmdct.txt    n_msg_bits.txt    提取出来的信息.txt


### 二、目录：
wav数据集：'D:\lt\数据集\dataset_128×1024（1声道，16 Khz, 32位，8s）'
概率图：1500_pro
载体mdct：2000_Qmdct_cover


### tips：
	1.  .m --> .exe
	`mcc   -m   lt_emb_STC`

 
