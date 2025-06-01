class Data_cleaning:

    def remove_outliers(self, data, columns):
        for column in columns:
            # calculate IQR to find outliers
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # remove outliers from the data
            data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        return data