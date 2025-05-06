import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataFile = pd.read_csv('data.csv')

independents = ['Age','Education_Level','Occupation','Number_of_Dependents','Location','Work_Experience','Marital_Status','Employment_Status','Household_Size','Homeownership_Status','Type_of_Housing','Gender','Primary_Mode_of_Transportation']
dependent = 'Income'

for i in range(len(independents)):
    print(dependent + ' vs ' + independents[i] + " Box\n")
    grouped = dataFile.groupby(independents[i])[dependent]
    independent = sorted(dataFile[independents[i]].unique())
    income_by_independent = [grouped.get_group(j) for j in independent]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.boxplot(income_by_independent, vert=True, patch_artist=True, tick_labels=independent, showfliers=False)
    ax.set_xlabel(independents[i])
    ax.set_ylabel(dependent)
    plt.tight_layout()
    plt.savefig(dependent + '_vs_' + independents[i] + "_box.png")
    plt.cla()
    plt.close()

for i in range(len(independents)):
    print(dependent + ' vs ' + independents[i] + " Box Outliers\n")
    grouped = dataFile.groupby(independents[i])[dependent]
    independent = sorted(dataFile[independents[i]].unique())
    income_by_independent = [grouped.get_group(j) for j in independent]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.boxplot(income_by_independent, vert=True, patch_artist=True, tick_labels=independent, showfliers=True)
    ax.set_xlabel(independents[i])
    ax.set_ylabel(dependent)
    plt.tight_layout()
    plt.savefig(dependent + '_vs_' + independents[i] + "_box_outliers.png")
    plt.cla()
    plt.close()

for i in range(len(independents)):
    print(dependent + ' vs ' + independents[i] + " Median Bar\n")
    medians = dataFile.groupby(independents[i])[dependent].median().reset_index()
    plt.bar(medians[independents[i]], medians[dependent])
    plt.yscale('linear')
    plt.xlabel(independents[i])
    plt.ylabel(dependent)
    plt.title(dependent + ' vs ' + independents[i])
    plt.savefig(dependent + '_vs_' + independents[i] + "_median_bar.png")
    plt.cla()
    plt.close()

for i in range(len(independents)):
    print(dependent + ' vs ' + independents[i] + " Median Plot\n")
    medians = dataFile.groupby(independents[i])[dependent].median().reset_index()
    plt.plot(medians[independents[i]], medians[dependent])
    plt.yscale('linear')
    plt.xlabel(independents[i])
    plt.ylabel(dependent)
    plt.title(dependent + ' vs ' + independents[i])
    plt.savefig(dependent + '_vs_' + independents[i] + "_median_plot.png")
    plt.cla()
    plt.close()

for i in range(len(independents)):
    print(dependent + ' vs ' + independents[i] + " Scatter\n")
    plt.scatter(dataFile[independents[i]], dataFile[dependent])
    plt.yscale('linear')
    plt.xlabel(independents[i])
    plt.ylabel(dependent)
    plt.title(dependent + ' vs ' + independents[i])
    plt.savefig(dependent + '_vs_' + independents[i] + "_scatter.png")
    plt.cla()
    plt.close()

for i in range(len(independents)):
    print(dependent + ' vs ' + independents[i] + " Bar\n")
    plt.bar(dataFile[independents[i]], dataFile[dependent])
    plt.yscale('linear')
    plt.xlabel(independents[i])
    plt.ylabel(dependent)
    plt.title(dependent + ' vs ' + independents[i])
    plt.savefig(dependent + '_vs_' + independents[i] + "_bar.png")
    plt.cla()
    plt.close()