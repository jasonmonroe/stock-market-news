# src/eda.py

# --- Exploratory Data Analysis --- #

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

from src import config

def labeled_barchart(df: pd.DataFrame, feature: str, perc: bool=False, n=None) -> None:
    """
    Barplot with percentage at the top
    """
    total = len(df[feature])
    count = df[feature].nunique()

    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)

    ax = sns.countplot(
        data=df,
        x=feature,
        palette="Paired",
        order=df[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc:
            label = "{:.1f}%".format(100 * p.get_height() / total)
        else:
            label = p.get_height()
        
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )
    plt.show()

def distribution_plot_wrt_target(data: pd.DataFrame, predictor: str, target: str) -> None:
    """
    Plots the distribution of a predictor variable with respect to a target variable.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # --- Plot 1: Overlaid Histogram ---
    axs[0].set_title(f'Distribution of {predictor.title()} by {target.title()}', fontsize=14, fontweight='bold')
    sns.histplot(
        data=data, x=predictor, kde=True, ax=axs[0], hue=target, palette='viridis', element='step'
    )
    
    # --- Plot 2: Boxplot with outliers ---
    axs[1].set_title(f'Boxplot of {predictor.title()} by {target.title()}', fontsize=14, fontweight='bold')
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1], palette='magma')

    # --- Plot 3: Boxplot without outliers ---
    axs[2].set_title(f'Boxplot of {predictor.title()} (No Outliers)', fontsize=14, fontweight='bold')
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[2], showfliers=False, palette='plasma')

    plt.tight_layout(pad=2.0)
    plt.show()

def histogram_boxplot(df: pd.DataFrame, feature: str, figsize=(14, 8), kde: bool = True, bins=None, style: str='seaborn-v0_8-whitegrid') -> None:
    """
    Enhanced boxplot and histogram combined.
    """
    plt.style.use(style)
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2, sharex=True, gridspec_kw={"height_ratios": (0.25, 0.75)}, figsize=figsize
    )

    f2.suptitle(f"Distribution of {feature.title()} with Boxplot and Histogram", fontsize=16, fontweight='bold', color='#333333')

    # Boxplot
    sns.boxplot(
        data=df, x=feature, ax=ax_box2, showmeans=True, color="#1f77b4",
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "red", "markersize": "8"},
        linewidth=1.5
    )
    ax_box2.set_xlabel("")
    ax_box2.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Histogram
    hist_color = "#2ca02c"
    if bins:
        sns.histplot(data=df, x=feature, kde=kde, ax=ax_hist2, bins=bins, color=hist_color)
    else:
        sns.histplot(data=df, x=feature, kde=kde, ax=ax_hist2, color=hist_color)

    mean_val = df[feature].mean()
    median_val = df[feature].median()

    ax_hist2.axvline(mean_val, color="#d62728", linestyle="--", linewidth=2.5, label=f"Mean: {mean_val:,.2f}")
    ax_hist2.axvline(median_val, color="#ff7f0e", linestyle="-", linewidth=2.5, label=f"Median: {median_val:,.2f}")

    ax_hist2.legend(loc='upper right', frameon=True, shadow=True, fancybox=True)
    ax_hist2.set_ylabel("Frequency / Count", fontsize=12)
    ax_hist2.set_xlabel(feature.title(), fontsize=12)
    ax_hist2.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    f2.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_word_count_distribution(data: pd.DataFrame, word_count_column: str) -> None:
    """
    Enhanced histogram for visualizing the distribution of word counts.
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    counts = data[word_count_column].dropna()
    mean_val = counts.mean()
    median_val = counts.median()

    plt.figure(figsize=(12, 7))
    sns.histplot(
        counts, bins=50, color='#3A5A40', edgecolor='#2a2a2a', kde=True, linewidth=0.8,
        line_kws={'color': '#A3B18A', 'linewidth': 3, 'alpha': 0.8}
    )

    plt.axvline(mean_val, color='#d62728', linestyle='--', linewidth=2.5, label=f"Mean: {mean_val:,.2f}")
    plt.axvline(median_val, color='#ff7f0e', linestyle=':', linewidth=2.5, label=f"Median: {median_val:,.2f}")

    column_str = word_count_column.replace('_', ' ').title()
    plt.title(f"Distribution of {column_str}", fontsize=18, fontweight='bold', pad=15)
    plt.xlabel(column_str, fontsize=14, labelpad=10)
    plt.ylabel('Frequency', fontsize=14, labelpad=10)
    plt.legend(loc='upper right', frameon=True, shadow=True, fancybox=True)
    sns.despine(trim=True)
    plt.tight_layout()
    plt.show()

def show_correlation_matrix(df: pd.DataFrame, matrix_title: str='Correlation Matrix') -> None:
    plt.figure(figsize=(14, 10))
    sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".3f", vmin=-1, vmax=1)
    plt.title(matrix_title.title(), fontsize=16)
    plt.tight_layout()
    plt.show() # In a script, this might block execution until closed

def show_plot_stock_price(df: pd.DataFrame) -> None:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Distribution of Stock Prices by News Sentiment', fontsize=16, y=1.02)
    axes = axes.flatten()

    price_variables = ['open', 'high', 'low', 'close']

    for i, variable in enumerate(price_variables):
        variable_title = variable.title()
        sns.boxplot(
            data=df, x="label_text", y=variable, ax=axes[i], palette='viridis',
            medianprops={'color': 'red', 'linewidth': 2}
        )
        axes[i].set_title(f'{variable_title} Price by Sentiment', fontsize=14)
        axes[i].set_xlabel('News Sentiment', fontsize=12)
        axes[i].set_ylabel(f'Price ({variable_title})', fontsize=12)

    plt.tight_layout(pad=3.0)
    plt.show()

def plot_top_word_freq(words, word_counts) -> None:
    """
    Enhanced horizontal bar chart for visualizing the Top N most frequent words.
    """
    style = 'seaborn-v0_8-white'
    palette = 'viridis_r'
    plt.style.use(style)

    # Combine into a DataFrame for easy sorting and handling
    df_words = pd.DataFrame({
        'Word': words,
        'Frequency': word_counts
    }).sort_values(by='Frequency', ascending=False).head(config.COMMON_WORD_CNT)

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        x='Frequency', y='Word', data=df_words, palette=palette, edgecolor='#333333', linewidth=0.5
    )

    for p in ax.patches:
        width = p.get_width()
        ax.annotate(
            f'{int(width):,}',
            (width, p.get_y() + p.get_height() / 2.),
            ha='left', va='center', xytext=(5, 0), textcoords='offset points',
            fontsize=11, color='black', fontweight='bold'
        )

    plt.title(f"Top {config.COMMON_WORD_CNT} Most Frequent Words", fontsize=18, fontweight='bold', pad=15)
    plt.xlabel("Word Frequency", fontsize=14, labelpad=10)
    plt.ylabel("Word", fontsize=14, labelpad=10)

    max_freq = df_words['Frequency'].max()
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels([f'{int(x):,}' for x in ax.get_xticks()])
    sns.despine(left=True, bottom=True)
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()