import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
  
# 读取CSV文件  
df = pd.read_csv('weight_exp_instruct.csv')  
  
# 将 "Loss" 列中的比值转换为分母的数值  
def loss_to_denominator(loss_str):  
    return float(loss_str.split(':')[1])  
  
df['DPO Loss'] = df['DPO Loss'].apply(loss_to_denominator)  
  
# 设置Seaborn风格  
sns.set(style="whitegrid")  
  
# 设置图表大小  
plt.figure(figsize=(10, 6))  
  
# 绘制各列随 Loss 分母变化的曲线  
columns_to_plot = ['TA Text', 'TA Image']  
markers = ['o', 's']  # 定义每条线的标记样式  
colors = ['#1f77b4', '#ff7f0e']  # 定义每条线的颜色  
  
for column, marker, color in zip(columns_to_plot, markers, colors):  
    plt.plot(df['DPO Loss'], df[column], marker=marker, label=column, color=color, linestyle='-', linewidth=2, markersize=8)  
  
# 设置图表标题和标签  
plt.xlabel('w2:w1 Ratio', fontsize=14)  # 改进后的x轴标签  
plt.ylabel('TA Score', fontsize=14)  # 改进后的y轴标签  
plt.title('TA Scores vs w2:w1 Ratio', fontsize=16)  # 改进后的标题  
plt.legend(fontsize=12)  
plt.grid(True, linestyle='--', alpha=0.7)  
  
# 设置刻度字体大小  
plt.xticks(fontsize=12)  
plt.yticks(fontsize=12)  
  
# 保存图表为高分辨率的PDF文件  
plt.savefig('ta_scores_vs_w2_w1_ratio.pdf', dpi=1000, bbox_inches='tight')  
  
# 显示图表  
plt.show()  
