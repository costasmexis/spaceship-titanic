
'''
Split TRAIN data to SUB-TRAIN and VALIDATION, in order to test models accuracy
Also, it creates the X_train.csv and y_train.csv (The WHOLE training set)
And the X_test.csv (The WHOLE test set)

Input should be data_train.csv and data_test.csv (the output of data_prepare.py)
Output would be X_sub_train.csv, y_sub_train.csv and X_val.csv, y_val.csv and X_test.csv
'''

# ===========================
# Import Libraries
# ===========================
import numpy as np
import pandas as pd
import argparse

from sklearn.model_selection import train_test_split

def main():

	def load_dataset(filename, train_or_test):
	    df = pd.read_csv(filename)
	    df = df.set_index('PassengerId')
	    if(train_or_test == 'train'):
	        df['Transported'] = df['Transported'].astype(int)
	        X = df.drop(['Transported'],axis=1)
	        y = df['Transported']
	        return X, y
	    else:
	        return df


	X, y = load_dataset("../data/data_train.csv", "train")
	X_test = load_dataset("../data/data_test.csv", "test")

	# =====================
	# Split train / val 
	# =====================
	# Train-validation split
	test_size = 0.2
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=0)
	print("Validation size is: ", test_size, "\n")


	X.to_csv("../data/X_train.csv", index=True)
	y.to_csv("../data/y_train.csv", index=True)

	X_test.to_csv("../data/X_test.csv", index=True)

	X_train.to_csv("../data/X_sub_train.csv", index=True)
	y_train.to_csv("../data/y_sub_train.csv", index=True)

	X_val.to_csv("../data/X_val.csv", index=True)
	y_val.to_csv("../data/y_val.csv", index=True)

if __name__ == '__main__':
	main()
