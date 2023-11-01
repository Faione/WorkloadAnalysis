import matplotlib.pyplot as plt
import seaborn as sns
import math


def plt_by_column(data, x_column="", columns=[], ncols=0):
    # 逐列绘图
    if columns == []:
      columns = data.columns
      
    total_plots = len(columns)
    nrows = 0
    if ncols != 0:
      nrows = total_plots // ncols + (1 if total_plots % ncols else 0)
    else:
      ncols = math.ceil(math.sqrt(total_plots))
      nrows = math.ceil(total_plots / ncols) 

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 4 * nrows))
    # 默认不进行绘制
    for ax in axes.flat:
      ax.axis('off')

    for i, column in enumerate(columns):
        row = i // ncols
        col = i % ncols
        axes[row, col].axis('on')
        if x_column != "":
          axes[row, col].plot(data[x_column], data[column], label=column)
          axes[row, col].set_xlabel(x_column)
        else:
          axes[row, col].plot(data[column], label=column)
          axes[row, col].set_xlabel(data.index.name)
          
        axes[row, col].legend()
        axes[row, col].set_title(column)
        axes[row, col].tick_params(axis='x', rotation=45)
      
    plt.tight_layout()
    plt.show()
  

def plt_corr_heatmap(corr_matrix):
    # 使用 seaborn 绘制热力图
    plt.figure(figsize=(80, 60)) # 可以调整大小
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')  # 'annot=True' 显示相关系数
    
    # 显示图形
    plt.show() 