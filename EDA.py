import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataFile = pd.read_csv('data.csv')

independents = ['Age','Education_Level','Occupation','Number_of_Dependents','Location','Work_Experience','Marital_Status','Employment_Status','Household_Size','Homeownership_Status','Type_of_Housing','Gender','Primary_Mode_of_Transportation']
dependent = 'Income'

for i in range(len(independents)):
    medians = dataFile.groupby(independents[i])[dependent].median()
    print(dependent + ' vs ' + independents[i] + '\n')
    plt.bar(medians.index, medians.values)
    plt.yscale('linear')
    plt.xlabel(independents[i])
    plt.ylabel(dependent)
    plt.title(dependent + ' vs ' + independents[i])
    plt.savefig(dependent + '_vs_' + independents[i] + "_median_bar.png")
    plt.cla()

for i in range(len(independents)):
    medians = dataFile.groupby(independents[i])[dependent].median()
    print(dependent + ' vs ' + independents[i] + '\n')
    plt.plot(medians.index, medians.values)
    plt.yscale('linear')
    plt.xlabel(independents[i])
    plt.ylabel(dependent)
    plt.title(dependent + ' vs ' + independents[i])
    plt.savefig(dependent + '_vs_' + independents[i] + "_median_plot.png")
    plt.cla()

for i in range(len(independents)):
    print(dependent + ' vs ' + independents[i] + '\n')
    plt.scatter(dataFile[independents[i]], dataFile[dependent])
    plt.yscale('linear')
    plt.xlabel(independents[i])
    plt.ylabel(dependent)
    plt.title(dependent + ' vs ' + independents[i])
    plt.savefig(dependent + '_vs_' + independents[i] + "_scatter.png")
    plt.cla()

for i in range(len(independents)):
    print(dependent + ' vs ' + independents[i] + '\n')
    plt.bar(dataFile[independents[i]], dataFile[dependent])
    plt.yscale('linear')
    plt.xlabel(independents[i])
    plt.ylabel(dependent)
    plt.title(dependent + ' vs ' + independents[i])
    plt.savefig(dependent + '_vs_' + independents[i] + "_bar.png")
    plt.cla()
