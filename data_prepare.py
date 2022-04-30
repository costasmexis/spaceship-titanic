import numpy as np
import pandas as pd
import argparse

def main():

	parser = argparse.ArgumentParser(description='Args for preparing dataset for dfing')
	parser.add_argument('-input','--input', help='Dataset (df or test)', required=True)
	parser.add_argument('-output','--output', help='Dataset ready for training', required=True)
	args = parser.parse_args()

	df = pd.read_csv(args.input)

	# on df set
	duplicate = df[df.duplicated()]

	if duplicate.empty:
		print('No duplicate rows!')
	else:
		print("Duplicate Rows :")
		duplicate

    # Checking the total amount of empty values in the dataset
	print(f' Missing Data: {df.isnull().sum().sum(): 5,.0f} Values \n')

	# Looking at NaN % within the df data
	nan = pd.DataFrame(df.isna().sum(), columns = ['NaN_sum'])
	nan['Percentage(%)'] = (nan['NaN_sum']/len(df))*100
	# nan['Type'] = nan.index.dtype()
	nan = nan[nan['NaN_sum'] > 0]
	nan = nan.sort_values(by = ['NaN_sum'])
	types = []
	for i in nan.index: 
	    types.append(df[i].dtype)
	nan['Type'] = types
	print(nan)

	# Create Age_group feature
	df['Age_group'] = np.nan
	df.loc[df['Age']<= 18, 'Age_group'] = 'Age_0-18'
	df.loc[(df['Age'] > 18) & (df['Age'] <= 36), 'Age_group'] = 'Age_19-36'
	df.loc[(df['Age'] > 36) & (df['Age'] <= 50), 'Age_group'] = 'Age_37-50'
	df.loc[df['Age'] > 50, 'Age_group'] = 'Age_50+'

	# Create a feature with total expences of each passenger
	expences = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
	df['Total_expences'] = df['Expenditure'] = df[expences].sum(axis=1)

	# Create a variable (0/1) describing if the current passenger spend anything or not!
	df['No_spending'] = (df['Total_expences']==0).astype(int)
	df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
	df['Group_size'] = df['PassengerId'].apply(lambda x: x.split('_')[1]).astype(int)
	df['Solo'] = (df['Group_size'] == 1).astype(int)

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


	# Imputing the missing data
	# Looking at NaN % within the df data
	nan = pd.DataFrame(df.isna().sum(), columns = ['NaN_sum'])
	nan['Percentage(%)'] = (nan['NaN_sum']/len(df))*100
	# nan['Type'] = nan.index.dtype()
	nan = nan[nan['NaN_sum'] > 0]
	nan = nan.sort_values(by = ['NaN_sum'])
	types = []
	for i in nan.index: 
	    types.append(df[i].dtype)
	nan['Type'] = types
	print(nan)	

	# Filling missing CryoSleep
	df['CryoSleep'].fillna(df['No_spending'], inplace=True)
	print('CryoSleep missing values: ', df['CryoSleep'].isna().sum())

	# Filling missing HomePlanet
	for i in df[df['HomePlanet'].isnull()].index:
		if((df.iloc[i]['Solo']==1)  & (df.iloc[i]['CryoSleep']==0) & (df.iloc[i]['No_spending']==0)):
			df.loc[i, 'HomePlanet'] = 'Earth'

	for i in df[df['HomePlanet'].isnull()].index:
	    if((df.iloc[i]['Destination']=='TRAPPIST-1e')):
	        df.loc[i, 'HomePlanet'] = 'Mars'

	df['HomePlanet'] = df['HomePlanet'].fillna('Europa')
	print('HomePlanet missing values: ', df['HomePlanet'].isna().sum())

	# Filling missing Age
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
	    if((df.iloc[i]['HomePlanet']=='Europa')): 
	        df.loc[i, 'Destination'] = '55 Cancri e'

	df['Destination'] = df['Destination'].fillna(df['Destination'].mode()[0])
	print('Destination missing values: ', df['Destination'].isna().sum())

	# Filling Cabin_side
	# We will drop nans on Cabin_side in lack of better idea
	df.drop(index=df[df['Cabin_side'].isna()].index, inplace=True)

	print('Cabin_side missing values: ', df['Cabin_side'].isna().sum())

	# Filling Cabin_deck
	for i in df[df['Cabin_deck'].isnull()].index:
	    if((df.iloc[i]['No_spending']==1) & (df.iloc[i]['HomePlanet']=='Earth')): 
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

	imputer_cols = ["FoodCourt", "ShoppingMall", "Spa", "VRDeck" ,"RoomService"]
	imputer = SimpleImputer(strategy="median" )
	imputer.fit(df[imputer_cols])
	df[imputer_cols] = imputer.transform(df[imputer_cols])


	# Set index PassengerId
	df = df.set_index('PassengerId')

	# Encode boolean variables
	# label encoding 
	from sklearn.preprocessing import LabelEncoder

	label_cols = ["HomePlanet", "CryoSleep", "Destination", "Age_group", "Cabin_deck", "Cabin_side", "Group_size"]
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
	df.drop(['Name','VIP','Cabin_num', 'Group'],axis=1,inplace=True)

	print(df.shape)

	# Save Final .csv
	df.to_csv(args.output, index=True)

# if __name__ == “main”: is used to execute some code only if the file was run directly, and not imported.
if __name__ == '__main__':
	main()





