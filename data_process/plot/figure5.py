import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
import os  
  
# 读取 CSV 文件  
csv_filename = 'lengths_uf.csv'  
df = pd.read_csv(csv_filename)  
  
# 将所有列的数据合并成一个 Series  
all_data = pd.concat([df[column] for column in df.columns])  
  
# 计算所有数据的分位数、最大值、最小值和平均值  
quantiles = all_data.quantile([0.25, 0.5, 0.75])  
max_value = all_data.max()  
min_value = all_data.min()  
mean_value = all_data.mean()  
  
# # 打印统计数据  
# print("所有数据的各分位数：")  
# print(quantiles)  
# print(f"所有数据的最大值: {max_value}")  
# print(f"所有数据的最小值: {min_value}")  
# print(f"所有数据的平均值: {mean_value}")  
# print()  # 空行分隔  
  
# # 将统计数据写入文件  
# output_filename = 'statistics_uf.txt'  
# with open(output_filename, 'w') as f:  
#     f.write("所有数据的各分位数：\n")  
#     f.write(quantiles.to_string() + '\n')  
#     f.write(f"所有数据的最大值: {max_value}\n")  
#     f.write(f"所有数据的最小值: {min_value}\n")  
#     f.write(f"所有数据的平均值: {mean_value}\n")  
  
# 设置图表的风格  
sns.set(style="whitegrid")  
  
# 创建图表  
plt.figure(figsize=(10, 6))  
  
# 绘制所有数据的 token 长度分布图  
sns.histplot(all_data, bins=30, kde=True, color='b', edgecolor='black')  
  
# 添加标题和标签  
plt.title('Histogram of Token Lengths in UltraFeedback Data', fontsize=16, weight='bold')  
plt.xlabel('Token Length', fontsize=14)  
plt.ylabel('Count', fontsize=14)  
  
# 设置刻度字体大小  
plt.xticks(fontsize=12)  
plt.yticks(fontsize=12)  
  
# 添加网格  
plt.grid(True, linestyle='--', alpha=0.7)  
  
# 去除顶部和右侧的边框  
sns.despine()  
  
# 确保 plot 目录存在  
if not os.path.exists('plot'):  
    os.makedirs('plot')  
  
# 保存图表到 plot 目录  
plt.savefig('plot/combined_token_length_distribution_uf.pdf')  
  
# 显示图表  
plt.show()  
