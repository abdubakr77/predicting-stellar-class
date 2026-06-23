import matplotlib.pyplot as plt

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