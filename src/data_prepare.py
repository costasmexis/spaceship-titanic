'''
In this file, we prapare the raw data given by the competition for machine learning training.


To run:
python3 data_prepare.py -input ../data/train.csv -output ../data/data_train.csv -type trainset
python3 data_prepare.py -input ../data/test.csv -output ../data/data_test.csv -type testset

'''
import numpy as np
import pandas as pd
import argparse

def main():

	parser = argparse.ArgumentParser(description='Args for preparing dataset for dfing')
	parser.add_argument('-input','--input', help='Dataset (train or test)', required=True)
	parser.add_argument('-output','--output', help='Dataset ready for dfing', required=True)
	parser.add_argument('-type','--type', help='Is input the trainset or the testset?', required=True)
	args = parser.parse_args()

	df = pd.read_csv(args.input)

	# on df set
	duplicate = df[df.duplicated()]

	if duplicate.empty:
		print('No duplicate rows!')
	else:
		print("Duplicate Rows :")
		duplicate
		return

    # Checking the total amount of empty values in the dataset
	print(f' Missing Data on dataset: {df.isnull().sum().sum(): 5,.0f} Values \n')

	# Create Age_group feature
	df['Age_group'] = np.nan
	df.loc[df['Age']<= 18, 'Age_group'] = 'Age_0-18'
	df.loc[(df['Age'] > 18) & (df['Age'] <= 36), 'Age_group'] = 'Age_19-36'
	df.loc[(df['Age'] > 36) & (df['Age'] <= 50), 'Age_group'] = 'Age_37-50'
	df.loc[df['Age'] > 50, 'Age_group'] = 'Age_50+'

	# Create a feature with total expences of each passenger
	expences = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
	df['Total_expences'] = df[expences].sum(axis=1)

	# Create a variable (0/1) describing if the current passenger spend anything or not!
	df['No_spending'] = (df['Total_expences']==0).astype(int)

	# Extrack passenger's group information from PassengerId
	df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
	df['Group_size'] = df['PassengerId'].apply(lambda x: x.split('_')[1]).astype(int)
	df['Solo'] = (df['Group_size'] == 1).astype(int)

	# Cabin
	# Replace NaN's with outliers for now (so we can split feature)
	df['Cabin'].fillna('Z/9999/Z', inplace=True)
	df['Cabin_deck'] = df['Cabin'].apply(lambda x: x.split('/')[0])
	df['Cabin_num'] = df['Cabin'].apply(lambda x: x.split('/')[1]).astype(int)
	df['Cabin_side'] = df['Cabin'].apply(lambda x: x.split('/')[2])
	# Put Nan's back in (we will fill these later)
	df.loc[df['Cabin_deck']=='Z', 'Cabin_deck']=np.nan
	df.loc[df['Cabin_num']==9999, 'Cabin_num']=np.nan
	df.loc[df['Cabin_side']=='Z', 'Cabin_side']=np.nan
	# Drop Cabin (we don't need it anymore)
	df.drop('Cabin', axis=1, inplace=True)
	#Cabin_deck = 'T' must be an outliers. It will be replaced with nan.
	df.loc[df['Cabin_deck']=='T', 'Cabin_deck']=np.nan
	# Create Cabin_num_group feature
	df['Cabin_num_group_1'] = (df['Cabin_num'] <= 300).astype(int)
	df['Cabin_num_group_2'] = ((df['Cabin_num'] > 300) & (df['Cabin_num'] <= 700)).astype(int)
	df['Cabin_num_group_3'] = ((df['Cabin_num'] > 700) & (df['Cabin_num'] <= 1200)).astype(int)

	# Name
	# Fill nans with outliers in order to execute feature engineering
	df['Name'].fillna('NoName NoName', inplace=True)
	df['First Name'] = df['Name'].apply(lambda x: x.split()[0])
	df['Family Name'] = df['Name'].apply(lambda x: x.split()[1])
	# Create feature Family indicating if the passenger is travelling with his family.
	df['Family'] = np.nan
	families = pd.DataFrame(df['Family Name'].value_counts())
	families = families[families['Family Name']>1].index.values
	df.loc[df['Family Name'].isin(families), 'Family'] = 1
	df.loc[~df['Family Name'].isin(families), 'Family'] = 0
	# Replace Family = 0 if Name = NoName NoName
	noname = df[df['Family Name'] == 'NoName'].index.values
	df.loc[noname, 'Family'] = 0
	df['Family'] = df['Family'].astype(int)
	# Drop Name column
	df.drop(['Name'],axis=1,inplace=True)


	# Imputing the missing data

	# Filling missing CryoSleep
	df['CryoSleep'].fillna(df['No_spending'], inplace=True)
	df['CryoSleep'] = df['CryoSleep'].astype(int)
	print('CryoSleep missing values: ', df['CryoSleep'].isna().sum())

	# Filling missing HomePlanet
	df['Previous Family Name'] = df.sort_values(by=['Family Name']).groupby(['Family Name'])['Family Name'].shift(1)
	df['Previous HomePlanet'] = df.sort_values(by=['Family Name']).groupby(['Family Name'])['HomePlanet'].shift(1)
	df['HomePlanet'].fillna(df['Previous HomePlanet'], inplace=True)
	df.sort_values(by=['Family Name'])['HomePlanet'].fillna(df.sort_values(by=['Family Name'])['Previous HomePlanet'], inplace=True)
	df.loc[(df['Cabin_deck']=='A') & (df['HomePlanet'].isna()), 'HomePlanet'] = 'Europa'
	df.loc[(df['Cabin_deck']=='B') & (df['HomePlanet'].isna()), 'HomePlanet'] = 'Europa'
	df.loc[(df['Cabin_deck']=='C') & (df['HomePlanet'].isna()), 'HomePlanet'] = 'Europa'
	df.loc[(df['Cabin_deck']=='D') & (df['HomePlanet'].isna()), 'HomePlanet'] = 'Mars'
	df.loc[(df['Cabin_deck']=='E') & (df['HomePlanet'].isna()), 'HomePlanet'] = 'Earth'
	df.loc[(df['Cabin_deck']=='F') & (df['HomePlanet'].isna()), 'HomePlanet'] = 'Earth'
	df.loc[(df['Cabin_deck']=='G') & (df['HomePlanet'].isna()), 'HomePlanet'] = 'Earth'
	# Fill remaining missing HomePlanet with the mode (Earth)
	df['HomePlanet'].fillna(df['HomePlanet'].mode()[0], inplace=True)
	print('HomePlanet missing values: ', df['HomePlanet'].isna().sum())

	# Filling missing Age
	df.loc[(df['VIP']==0) & (df['Age'].isna()), 'Age'] = df[df['VIP']==0]['Age'].median()
	df.loc[(df['VIP']==1) & (df['Age'].isna()), 'Age'] = df[df['VIP']==1]['Age'].median()
	df['Age'] = df['Age'].fillna(df.groupby(['HomePlanet', 'Solo', 'No_spending'])['Age'].transform('median'))
	print('Age missing values: ', df['Age'].isna().sum())

	# Fill also Age_group
	df.loc[df['Age']<= 18, 'Age_group'] = 'Age_0-18'
	df.loc[(df['Age'] > 18) & (df['Age'] <= 36), 'Age_group'] = 'Age_19-36'
	df.loc[(df['Age'] > 36) & (df['Age'] <= 50), 'Age_group'] = 'Age_37-50'
	df.loc[df['Age'] > 50, 'Age_group'] = 'Age_50+'
	print('Age_group missing values: ', df['Age_group'].isna().sum())

	# Filling missing Destination

	for i in df[df['Destination'].isnull()].index:
	    if((df.iloc[i]['HomePlanet']=='Earth')): 
	        df.loc[i, 'Destination'] = 'TRAPPIST-1e'

	for i in df[df['Destination'].isnull()].index:
	    if((df.iloc[i]['HomePlanet']=='Mars')): 
	        df.loc[i, 'Destination'] = 'TRAPPIST-1e'
	        
	for i in df[df['Destination'].isnull()].index:
	    if((df.iloc[i]['HomePlanet']=='Europa')): 
	        df.loc[i, 'Destination'] = '55 Cancri e'
	for i in df[df['Destination'].isnull()].index:
		if((df.iloc[i]['HomePlanet']=='Europa')): 
			df.loc[i, 'Destination'] = '55 Cancri e'

	print('Destination missing values: ', df['Destination'].isna().sum())

	# Filling Cabin_side
	df.loc[(df['HomePlanet']=='Earth') & (df['Cabin_side'].isna()), 'Cabin_side'] = 'P'
	df.loc[(df['HomePlanet']=='Europa') & (df['Cabin_side'].isna()), 'Cabin_side'] = 'S'
	df.loc[(df['HomePlanet']=='Mars') & (df['Cabin_side'].isna()), 'Cabin_side'] = 'P'
	print('Cabin_side missing values: ', df['Cabin_side'].isna().sum())

	# Filling Cabin_deck
	for i in df[df['Cabin_deck'].isnull()].index:
		if((df.iloc[i]['No_spending']==1)): 
			df.loc[i, 'Cabin_deck'] = 'G'
		else: 
			df.loc[i, 'Cabin_deck'] = 'F'

	print('Cabin_deck missing values: ', df['Cabin_deck'].isna().sum())

	# Filling remaining missing values of numerical features
	df.loc[df['No_spending']==1, 'RoomService'] = 0
	df.loc[df['No_spending']==1, 'FoodCourt'] = 0
	df.loc[df['No_spending']==1, 'Spa'] = 0
	df.loc[df['No_spending']==1, 'ShoppingMall'] = 0

	from sklearn.impute import SimpleImputer

	# filling the null values with median 
	from pickle import dump, load

	imputer_cols = ["FoodCourt", "ShoppingMall", "Spa", "VRDeck" ,"RoomService"]

	if(args.type == 'trainset'):
		print('Imputing TRAIN SET: ')
		imputer = SimpleImputer(strategy="median" )
		imputer.fit(df[imputer_cols])
		# Saving imputer for using it for test set.
		dump(imputer, open('imputer.pkl', 'wb'))
		df[imputer_cols] = imputer.transform(df[imputer_cols])
	else:
		print('Imputing TEST SET')
		imputer = load(open('imputer.pkl', 'rb'))
		df[imputer_cols] = imputer.transform(df[imputer_cols])


	# Set index PassengerId
	df = df.set_index('PassengerId')

	# Encode boolean variables
	# label encoding 
	from sklearn.preprocessing import LabelEncoder

	label_cols = ["HomePlanet", "Destination", "Age_group", "Cabin_deck", "Group_size", "Cabin_side"]
	def label_encoder(d, columns):
	    for col in columns:
	        d[col] = d[col].astype(str)
	        d[col] = LabelEncoder().fit_transform(d[col])
	    return d

	df = label_encoder(df, label_cols)
	df.shape	

	for col in label_cols:
	    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
	    df = pd.concat([df, dummies], axis=1)
	    df.drop([col], axis=1, inplace=True)

	
	# df['Transported'] = df['Transported'].astype(int)

	# Drop unusefull columns
	df.drop(['First Name','Family Name', 'VIP','Cabin_num','Group', 'Previous Family Name', 'Previous HomePlanet'],axis=1,inplace=True)

	print(df.shape)

	# Save Final .csv
	df.to_csv(args.output, index=True)

# if __name__ == “main”: is used to execute some code only if the file was run directly, and not imported.
if __name__ == '__main__':
	main()





