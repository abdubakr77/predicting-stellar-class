import matplotlib.pyplot as plt
import seaborn as sns

def show_count_plots(df,col,draw_pie_chart = True):
    counts = df[col].value_counts()

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].barh(counts.index, counts.values, color='steelblue')
    ax[0].set_xlabel('Count')
    ax[0].set_title('Bar Chart Class Distribution')

    if draw_pie_chart:
        ax[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%')
        ax[1].set_title('Pie Chart Class Distribution')

    plt.tight_layout()
    plt.show()

def show_corr(df,save_path:str=None):
    plt.figure(figsize=(15, 12))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

    plt.title('Correlation Matrix')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def show_outliers(df , save_path:str=None):
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(df)
    plt.xticks(rotation=45)
    plt.grid()
    plt.title('Box Plot')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()