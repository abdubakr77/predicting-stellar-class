import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


def show_count_plots(df, col, draw_pie_chart=True, save_path: str = None):
    """
    Plot a bar chart and optional pie chart of value counts for a column.

    Args:
        df: dataframe containing the column
        col: column name to count
        draw_pie_chart: whether to also draw the pie chart
        save_path: if given, saves the figure to this path

    Returns:
        None, shows the plot
    """
    counts = df[col].value_counts()

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].barh(counts.index, counts.values, color='steelblue')
    ax[0].set_xlabel('Count')
    ax[0].set_title('Bar Chart Class Distribution')

    if draw_pie_chart:
        ax[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%')
        ax[1].set_title('Pie Chart Class Distribution')

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.tight_layout()
    plt.show()


def show_corr(df, save_path: str = None, h=15, w=12):
    """
    Plot a correlation heatmap for all numeric columns in a dataframe.

    Args:
        df: dataframe to correlate
        save_path: if given, saves the figure to this path
        h: figure height
        w: figure width

    Returns:
        None, shows the plot
    """
    plt.figure(figsize=(h, w))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)

    plt.title('Correlation Matrix')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def show_outliers(df, save_path: str = None):
    """
    Plot a boxplot across all columns to spot outliers at a glance.

    Args:
        df: dataframe to plot
        save_path: if given, saves the figure to this path

    Returns:
        None, shows the plot
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(df)
    plt.xticks(rotation=45)
    plt.grid()
    plt.title('Box Plot')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def show_confusion_matrix(y_true, y_pred, target_names=None, save_path: str = None):
    """
    Plot a confusion matrix as a heatmap.

    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        target_names: optional class names for the axis labels
        save_path: if given, saves the figure to this path

    Returns:
        None, shows the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=None if target_names is None else target_names,
                yticklabels=None if target_names is None else target_names)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: list = None,
    factor: float = 2.0) -> pd.DataFrame:
    """
    Remove outliers using the IQR method.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list
        Numerical columns to process.
    factor : float
        IQR multiplier (default=2.0).

    Returns
    -------
    pd.DataFrame
        DataFrame after removing outliers.
    """

    df = df.copy()

    if columns is None:
        columns = ["u", "g", "r", "i", "z"]

    mask = pd.Series(True, index=df.index)

    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        mask &= df[col].between(lower, upper)

    return df.loc[mask].reset_index(drop=True)


def plot_roc_curve(y_true, y_probas, target_names):
    """
    Plot a one-vs-rest ROC curve for each class.

    Args:
        y_true: encoded labels (0, 1, 2)
        y_probas: predict_proba output (n_samples, n_classes)
        target_names: class names matching the encoded labels, in order

    Returns:
        None, shows the plot
    """
    n_classes = len(target_names)

    # one-hot the labels so each class can be scored against the rest
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure(figsize=(10, 6))

    colors = ['blue', 'orange', 'green']

    for i, (name, color) in enumerate(zip(target_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probas[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color,
                 label=f'{name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()