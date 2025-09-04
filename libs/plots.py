import seaborn as sns
from matplotlib import pyplot as plt

def make_boxplot_single(df, 
        colors, 
        cols:int=3,
        y=None,
        hue=None,
        columns:list=[],
        export:bool=False,
        name_export:str="plot.png"):
    
    fig, ax = plt.subplots(1,cols, figsize=(10,6))

    for i in range(len(columns)):
        sns.boxplot(
            ax=ax[i],
            data=df,
            x=columns[i],
            y=y,
            hue=hue,
            fill=False,
            palette=colors
        )
    
    sns.despine()

    if export:
        plt.savefig(name_export, dpi=300)

def make_plot(
        df, 
        colors, 
        rows:int=2,
        cols:int=3,
        y=None,
        hue=None,
        columns:list=[],
        plot_type:str="boxplot",
        export:bool=False,
        name_export:str="plot.png"):

    fig, ax = plt.subplots(rows,cols, figsize=(10,6))

    index = 1
    i=0
    j=0

    for column in columns:
        
        if index == len(columns):

            if plot_type == "boxplot":
                sns.boxplot(
                    ax=ax[i,j] if rows>1 else ax[j],
                    data=df,
                    x=column,
                    y=y,
                    hue=hue,
                    fill=False,
                    palette=colors
                )
            else:
                sns.histplot(
                    ax=ax[i,j] if rows>1 else ax[j],
                    data=df,
                    palette=colors,
                    x=column,
                    hue=hue,
                    fill=False,
                    kde=True
                )
        else:

            if plot_type == "boxplot":
                sns.boxplot(
                    ax=ax[i,j] if rows>1 else ax[j],
                    data=df,
                    x=column,
                    hue=hue,
                    y=y,
                    fill=False,
                    palette=colors,
                    legend=False
                )
            else:
                sns.histplot(
                    ax=ax[i,j] if rows>1 else ax[j],
                    data=df,
                    palette=colors,
                    x=column,
                    hue=hue,
                    fill=False,
                    kde=True,
                    legend=False
                )
        index+=1

        if index == cols:
            i+=1
            j=0
        else:
            j+=1

    sns.despine()

    if export:
        plt.savefig(name_export, dpi=300)