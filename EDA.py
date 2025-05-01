import matplotlib.pyplot as plt
import pandas as pd

dataFile = pd.read_csv('data.csv')

i = 0;

independents = ['Age','Education_Level','Occupation','Number_of_Dependents','Location','Work_Experience','Marital_Status','Employment_Status','Household_Size','Homeownership_Status','Type_of_Housing','Gender','Primary_Mode_of_Transportation']
dependent = 'Income'

plt.plot(df[independents[i]], df[dependent])
plt.xlabel(independents[i])
plt.ylabel(dependent)
plt.title(dependent + ' vs ' + independent[i])
plt.show()
