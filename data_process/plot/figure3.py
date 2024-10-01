import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import os  
  
def read_lengths_from_csv(filename='lengths_difference.csv'):  
    df = pd.read_csv(filename)  
    return df['response_lengths'].tolist(), df['revised_text_lengths'].tolist()  
  
def calculate_statistics(response_lengths, revised_text_lengths):  
    differences = np.array(revised_text_lengths) - np.array(response_lengths)  
      
    # 计算各分位数  
    percentiles = np.percentile(differences, [0, 25, 50, 75, 100])  
      
    # 最大最小值  
    max_difference = np.max(differences)  
    min_difference = np.min(differences)  
      
    # 统计信息  
    statistics = {  
        'min': min_difference,  
        '25th_percentile': percentiles[1],  
        'median': percentiles[2],  
        '75th_percentile': percentiles[3],  
        'max': max_difference  
    }  
      
    return differences, statistics  
  
def save_statistics(statistics, output_path='plot/statistics_difference.txt'):  
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  
    with open(output_path, 'w') as f:  
        for key, value in statistics.items():  
            f.write(f"{key}: {value}\n")  
  
def plot_difference_distribution(differences, output_path='plot/difference_distribution.pdf'):  
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  
      
    # 设置 LaTeX 字体  
    plt.rc('font', family='serif')  
    plt.rc('text', usetex=True)  
      
    plt.figure(figsize=(10, 6))  
    plt.hist(differences, bins=100, alpha=0.75, color='blue', edgecolor='black')  
      
    # plt.title('Distribution of Length Differences', fontsize=16)  
    plt.xlabel('Revised Text Length - Chosen Text Length', fontsize=14)  
    plt.ylabel('Count', fontsize=14)  
    plt.grid(True, linestyle='--', linewidth=0.5)  
      
    plt.tight_layout()  
    plt.savefig(output_path, dpi=300)  
    plt.show()  
  
# 主函数  
def main():  
    response_lengths, revised_text_lengths = read_lengths_from_csv()  
    differences, statistics = calculate_statistics(response_lengths, revised_text_lengths)  
    # save_statistics(statistics)  
    plot_difference_distribution(differences)  
  
main()  
