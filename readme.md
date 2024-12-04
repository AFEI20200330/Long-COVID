1. 2024.12.3 14:55 按照功能对蛋白质进行分类（未完成转2）
2. 2024年12月3日15:18:22 爬虫 所有的蛋白质功能  uniprot.py  
3. 2024年12月3日18:30:19 得到output共有4458个不同UniProt,源文件是有4511个不同的target
4. 对这4458个UniProt进行分类(output.xlsx)
Structural：结构性蛋白，指的是那些主要提供细胞或组织结构支持的蛋白质。
Enzyme：酶类蛋白，这类蛋白质能催化生物化学反应。
Transcription Factor：转录因子，这类蛋白质主要参与调控基因的表达。
Signal Transduction：信号传导，涉及传递和放大细胞内外的信号。
Immune System：免疫系统相关蛋白，参与身体的免疫反应。
Transport：运输蛋白，负责物质的运输和跨膜转运。
Cell Cycle：细胞周期相关蛋白，参与细胞生长、分裂的调控。
Apoptosis：凋亡相关蛋白，参与调控细胞的程序性死亡。
Metabolic Process：代谢过程相关蛋白，参与细胞的代谢活动。
Others：其他类别，用于归类那些不易直接归入上述任何一个类别的蛋白质。

5. 计算了不同功能类别在每个聚类中蛋白质的平均表达水平。下面是不同聚类的蛋白质功能类别平均表达情况：
功能类别         	      聚类 0 的平均表达水平	聚类 1 的平均表达水平
凋亡（Apoptosis）	       10.79	              10.80
细胞周期（Cell Cycle）       10.78	              10.80
酶（Enzyme）     	       10.65	              10.65
免疫系统（Immune System）	   10.88        	      10.90
代谢过程（Metabolic Process）10.31	              10.30
其他（Others）	           10.54	              10.54
信号传导（Signal Transduction）10.57	              10.58
结构性（Structural）	       10.48	              10.50
转录因子（Transcription Factor）10.45	              10.44
运输（Transport）	        10.36	              10.37
尽管两个聚类之间的蛋白质表达差异不大，但某些功能类别（如免疫系统和细胞周期）在聚类 1 中略微高于聚类 0。这些差异可能反映了不同聚类中生物学过程的微妙变化。

7. 分析每种蛋白，在0和1类别之间的差异显著性，Mann-Whitney Utest，因为有蛋白表现为非正态性。（应尝试多种差异性分析方法）

8. 2024年12月4日13:58:47  使用差异显著的蛋白，MLP_attention:
Test Accuracy: 0.6500
Confusion Matrix:
 [[12  8]
 [ 6 14]]
Precision: 0.6515
Recall: 0.6500
F1 Score: 0.6491
ROC AUC: 0.6925
9.  2024年12月4日14:14:40 根据二分类cluster进行临床特征可视化(在cluster/twucluster_clinical/res)

10. 开始对7尝试多种差异分析方法(服从正态分布的使用t-test，不服从的使用manu)
结果有157个蛋白,并完成了可视化

11. 根据二分类对临床特征进行差异分析()

12. 2024年12月4日17:28:38将四聚类标签作为特征数据

13.分析不同的cluster中updated——cluster的分布
![img_1.png](img_1.png)

