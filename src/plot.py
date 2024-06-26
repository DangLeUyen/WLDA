from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict
from funcs import convert_cov_to_corr, calculate_SE, calculate_diference

############ SHAPLEY VALUES ############

def aggregate_shap_class(expldf, pred_y):
    p = expldf[0].shape[0]
    class_values = np.unique(pred_y)
    shapley_values = pd.DataFrame(columns=np.arange(p), index=class_values)
    for c in class_values:
        idx = np.where(pred_y == c)[0]
        class_shap = pd.DataFrame(index = idx, columns = class_values)
        temp = []
        for j in range(p):
            for i in idx:
                class_shap.loc[i] = expldf[i][j]
            temp.append(np.abs(class_shap).mean(axis=0)[c])
        shapley_values.loc[c] = temp
    return shapley_values

def shapley_feature_importance(res, pred, colors, column_names, algs):

    # Create a figure with subplots
    fig, axs = plt.subplots(ncols=6,nrows=5, figsize=(18, 15), sharey=True, sharex=True)
    fig.patch.set_facecolor('#F1F1F1')

    #sns.set(rc = {'axes.facecolor': '#F1F1F1', 'figure.facecolor': '#FFFFFF'})
    # Plot each correlation matrix on the appropriate subplot
    for row, missing_rate in enumerate(res.keys()):
        matrix = res[missing_rate]
        predicteclabels = pred[missing_rate]
        df = pd.DataFrame(index = algs, columns=['class 0','class 1', 'class 2'])
        for col, model in enumerate(matrix.keys()):
            df = aggregate_shap_class(matrix[model], predicteclabels[model])
            df.columns = column_names
            df = df.reset_index(names='Label')
            df['Label'] = 'Class ' + df['Label'].astype(int).astype(str)
            df_melted = df.melt(id_vars=['Label'], var_name='Feature', value_name='Value')
            
            palette = colors
            sns.set()
            
            im = sns.barplot(y = 'Feature', x='Value', hue='Label', data=df_melted, ax=axs[row, col], 
                             orient='h', palette=palette, saturation=100, width=0.8 )
            axs[row, col].legend_.remove()
            axs[row, col].set(xlabel=None)
            axs[row, col].patch.set_facecolor('white') 
            axs[row, col].set_xticks([0, 0.3, 0.6])
            axs[row, col].set_xticklabels([0, 0.3, 0.6])
            axs[row, col].grid(True, which='both', axis='x', color='#FDF4F5')
            #axs[row, col].grid(True, which='both', axis='y', color='#FDF4F5')
            axs[row, col].xaxis.grid(True)
            for yline in [0.5, 1.5, 2.5]:
                axs[row, col].axhline(yline, color='#FFD5E5', linewidth=0.5)
            

        axs[row,0].set_ylabel(f'{missing_rate*100}%', rotation='horizontal', labelpad = 30)
        axs[4, 2].set_xlabel('Level of contribution', labelpad = 20, fontsize=14)

    for j in range(len(algs)):
        axs[0, j].set_title(f'{algs[j]}', fontsize=14)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines[:3],labels[:3], loc='lower center', ncol = len(labels), bbox_to_anchor=(0.15,0.06))
    
    # save pic
    plt.savefig(f'results/plots/shapleyvalues/barplot.png',dpi=300, bbox_inches='tight')
    # plot pic
    
    #plt.show()

def shapley_heatmap(res, colors, column_names, algs, label='class 0'):
    cmap = LinearSegmentedColormap.from_list('', colors)

    sns.set(rc = {'axes.facecolor': '#F1F1F1', 'figure.facecolor': '#F1F1F1'})
    # Create a figure with subplots
    fig, axs = plt.subplots(ncols=6,nrows=5, figsize=(18, 15), sharex=True, sharey=True)
    xticks = np.arange(0, 150, 30)
    # Plot each correlation matrix on the appropriate subplot
    for row, missing_rate in enumerate(res.keys()):
        matrix = res[missing_rate]
        for col, model in enumerate(matrix.keys()):
            f0df = pd.DataFrame(columns=['class 0','class 1', 'class 2'])
            f1df = pd.DataFrame(columns=['class 0','class 1', 'class 2'])
            f2df = pd.DataFrame(columns=['class 0','class 1', 'class 2'])
            f3df = pd.DataFrame(columns=['class 0','class 1', 'class 2'])
            for i in range(30):        
                f0df.loc[i] = matrix[model][i][0].tolist()
                f1df.loc[i] = matrix[model][i][1].tolist()
                f2df.loc[i] = matrix[model][i][2].tolist()
                f3df.loc[i] = matrix[model][i][3].tolist()
            df = pd.concat([f0df.T.loc[label], f1df.T.loc[label], f2df.T.loc[label], f3df.T.loc[label]], axis=1)
            df.columns = column_names
            im = sns.heatmap(df.T, cmap=cmap, cbar=False, vmin=-0.5, vmax=0.5, ax=axs[row, col])
            im.set_xticks([0, 10, 20, 30]) # <--- set the ticks first
            im.set_xticklabels([0, 10, 20, 30], rotation=0)

        axs[row,0].set_ylabel(f'{missing_rate*100}%', rotation='horizontal', labelpad = 30)
    
    cbar_ax = fig.add_axes([.92, .3, .02, .4])
    fig.colorbar(im.get_children()[0], cax=cbar_ax)
    
    for j in range(len(algs)):
        axs[0, j].set_title(f'{algs[j]}', fontsize=14)
    
    # save pic
    plt.savefig(f'results/plots/shapleyvalues/{label}.png',dpi=300, bbox_inches='tight')
    # plot pic
    #plt.show()
    plt.close()

################# BOUNDARY DECISION COSINE SIMILARITY ###################
def boundary_barplot(df, missing_range):
    
    # Convert the DataFrame into a format suitable for plotting
    boundaries = df['boundary'].unique()
    markers = ['+', 's', '^', 'x', 'o', '+', 'D']

    # Plotting the lines
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.patch.set_facecolor('#F1F1F1')
    # Store handles and labels for the legend
    handles = []
    labels = []
    for i, boundary in enumerate(boundaries):
        ax = axes.flat[i]
        ax.set_facecolor('white')
        subset = df[df['boundary'] == boundary]
        for j, model in enumerate(subset['models']):
            line, = ax.plot(missing_range, subset[subset['models'] == model].iloc[0, 2:], label=model, marker=markers[j], markersize=6, linewidth=2)
            # Collect handles and labels for the first subplot
            if i == 0:
                handles.append(line)
                labels.append(model)
        ax.set_title(f'Boundary between class {boundary[0]} and class {boundary[1]}', fontsize=14)
        ax.set_xticks(missing_range)
        ax.set_ylim(0, 1.1)  # Set y-axis to start from 0

    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(-0.08, 0.5))
    fig.text(0.5, -0.02, 'Missing rate', ha='center', va='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(f'results/plots/boundarysimilary/boundary.png',dpi=300, bbox_inches='tight')
    plt.close()

###################### CORRELATION HEATMAPS ################
def all_mr_heatmaps(Ss, colors, t='correlation'):
    cmap = LinearSegmentedColormap.from_list('', colors)
    diff_matrix = defaultdict(list)
    corrmx = defaultdict(list)

    sns.set(rc = {'axes.facecolor': '#F1F1F1', 'figure.facecolor': '#F1F1F1'})
    # Create a figure with subplots
    fig, axs = plt.subplots(ncols=6,nrows=5, figsize=(21, 15))
    
    # Plot each correlation matrix on the appropriate subplot
    for row, missing_rate in enumerate(Ss.keys()):
        matrix = Ss[missing_rate][0]
        for col in range(len(matrix)):
            corrmx[missing_rate].append(convert_cov_to_corr(matrix[col]))
            if t == 'mse_corr':
                mse_mx = calculate_SE(corrmx[missing_rate][0], corrmx[missing_rate][col])
                diff_matrix[missing_rate].append(mse_mx)
                vmax=0.4
            elif t == 'sub_corr':
                sub_mx = calculate_diference(corrmx[missing_rate][0], corrmx[missing_rate][col])
                diff_matrix[missing_rate].append(sub_mx)
            else:
                diff_matrix[missing_rate] = corrmx[missing_rate]
                
            # Create a mask for the upper triangle
            mask = np.tril(np.ones_like(diff_matrix[missing_rate][0], dtype=bool), k=-1)
            if t == 'mse_corr':
                im = sns.heatmap(diff_matrix[missing_rate][col], cmap=cmap, cbar=False, ax=axs[row, col],vmax=vmax, mask=mask)
            else:
                im = sns.heatmap(diff_matrix[missing_rate][col], cmap=cmap, cbar=False, ax=axs[row, col],vmin=-1, vmax=1, mask=mask)
        
        axs[row,0].set_ylabel(f'{missing_rate*100}%', rotation='horizontal')
    
    cbar_ax = fig.add_axes([.92, .3, .02, .4])
    fig.colorbar(im.get_children()[0], cax=cbar_ax)

    algs = ["Ground Truth", "WLDA", "KNNI", "MICE", "Soft-Impute", "DIMV"]
    for j in range(len(algs)):
        axs[0, j].set_title(f'{algs[j]}', fontsize=14)

    title = 'Correlation Heatmaps'
    if t == 'mse_corr':
        title = 'Local MSE Difference Heatmaps for Correlation'
    elif t == 'sub_corr':
        title = 'Local Difference (Matrix Subtraction) Heatmaps for Correlation'
        # Add a title for the entire figure
    #fig.suptitle(f'{title}', fontsize=16)

        # Adjust the spacing between subplots
    #plt.subplots_adjust(wspace=0.1, hspace=0.1)
        # Hide the ticks and labels on all subplots
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
    # save pic
    plt.savefig(f'results/plots/correlation/{t}.png',dpi=300, bbox_inches='tight')
    # plot pic
    #plt.show()
    plt.close()

def each_mr_heatmaps(Ss, colors, t='correlation'):
    cmap = LinearSegmentedColormap.from_list('', colors)
    diff_matrix = defaultdict(list)
    corrmx = defaultdict(list)

    sns.set(rc = {'axes.facecolor': '#F1F1F1', 'figure.facecolor': '#F1F1F1'})
    
    # Plot each correlation matrix on the appropriate subplot
    for missing_rate in Ss.keys():
        # Create a figure with subplots
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
        algs = ["Ground Truth", "WLDA", "KNNI", "MICE", "Soft-Impute", "DIMV"]
        matrix = Ss[missing_rate][0]
        for i, ax in enumerate(axs.flat):
            if i < len(matrix):
                corrmx[missing_rate].append(convert_cov_to_corr(matrix[i]))
                if t == 'mse_corr':
                    mse_mx = calculate_SE(corrmx[missing_rate][0], corrmx[missing_rate][i])
                    diff_matrix[missing_rate].append(mse_mx)
                    vmax=0.4
                elif t == 'sub_corr':
                    sub_mx = calculate_diference(corrmx[missing_rate][0], corrmx[missing_rate][i])
                    diff_matrix[missing_rate].append(sub_mx)
                else:
                    diff_matrix[missing_rate] = corrmx[missing_rate]
                    
                # Create a mask for the upper triangle
                mask = np.tril(np.ones_like(diff_matrix[missing_rate][0], dtype=bool), k=-1)
                if t == 'mse_corr':
                    im = sns.heatmap(diff_matrix[missing_rate][i], cmap=cmap, cbar=False, ax=ax,vmax=vmax, mask=mask)
                else:
                    im = sns.heatmap(diff_matrix[missing_rate][i], cmap=cmap, cbar=False, ax=ax,vmin=-1, vmax=1, mask=mask)
                #sns.heatmap(matrix[i], ax=ax, cmap='viridis', cbar=True)
                ax.set_title(algs[i])
            else:
                ax.axis('off')  # Turn off the axis for empty subplots
        
    
        cbar_ax = fig.add_axes([.92, .3, .02, .4])
        fig.colorbar(im.get_children()[0], cax=cbar_ax)


        title = 'Correlation Heatmaps'
        if t == 'mse_corr':
            title = 'Local MSE Difference Heatmaps for Correlation'
        elif t == 'sub_corr':
            title = 'Local Difference (Matrix Subtraction) Heatmaps for Correlation'
            # Add a title for the entire figure
        #fig.suptitle(f'{title}', fontsize=16)

            # Adjust the spacing between subplots
        #plt.subplots_adjust(wspace=0.1, hspace=0.1)
            # Hide the ticks and labels on all subplots
        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            
        # save pic
        plt.savefig(f'results/plots/correlation/{int(missing_rate*100)}/{t}.png',dpi=300, bbox_inches='tight')
        # plot pic
        #plt.show()
        plt.close()
