'''
For more info read coursera discussion forum: https://www.coursera.org/learn/competitive-data-science/discussions/weeks/3
'''



############################ Normal (Leads to overfitting)

##### Method 1
# Calculate a mapping: {item_id: target_mean}
item_id_target_mean = all_data.groupby('item_id').target.mean()

# In our non-regularized case we just *map* the computed means to the `item_id`'s
all_data['item_target_enc'] = all_data['item_id'].map(item_id_target_mean)

# Fill NaNs
all_data['item_target_enc'].fillna(0.3343, inplace=True) 


##### Method 2

'''
     Differently to `.target.mean()` function `transform` 
   will return a dataframe with an index like in `all_data`.
   Basically this single line of code is equivalent to the first two lines from of Method 1.
'''
all_data['item_target_enc'] = all_data.groupby('item_id')['target'].transform('mean')

# Fill NaNs
all_data['item_target_enc'].fillna(0.3343, inplace=True) 




############################ K-fold mean encoding (Info leak still present but less than leave one out)

kf = model_selection.KFold(n_splits = 5, shuffle = False) 
all_data['item_target_enc'] = np.nan

for tr_ind, val_ind in kf.split(all_data):
    X_tr, X_val = all_data.iloc[tr_ind], all_data.iloc[val_ind]
    X_val['item_target_enc'] =   X_val['item_id'].map(X_tr.groupby('item_id').target.mean())
    all_data.iloc[val_ind,:] = X_val
    

all_data['item_target_enc'].fillna(0.3343, inplace = True)
encoded_feature = all_data['item_target_enc'].values


############################ Leave-one-out scheme (More info leak than k-fold)

leave_one_out_sum = all_data['item_id'].map(all_data.groupby('item_id').target.sum())
leave_one_out_count = all_data['item_id'].map(all_data.groupby('item_id').target.count())
all_data['item_target_enc'] = (leave_one_out_sum - all_data['target']) / (leave_one_out_count - 1)

all_data['item_target_enc'].fillna(0.3343, inplace = True)
encoded_feature = all_data['item_target_enc'].values


############################ Smoothing (Need to tune parameter alpha, less overfitting than previous cases)

alpha = 100
globalmean = 0.3343
train_new = all_data.copy()
nrows = train_new.groupby('item_id').size()

# all_data['item_id_target_mean'] = all_data.groupby('item_id')['target'].transform('mean')

means = train_new.groupby('item_id').target.agg('mean')

score = (np.multiply(means,nrows) + globalmean*alpha) / (nrows+alpha)

train_new['smooth'] = train_new['item_id']
train_new['smooth'] = train_new['smooth'].map(score)

encoded_feature = train_new['smooth'].values


############################ Expanding mean scheme (No info leak at all, used as default feature in catboost)

cumsum = all_data.groupby('item_id')['target'].cumsum() - all_data['target']
cumcnt = all_data.groupby('item_id').cumcount()

all_data['item_target_enc_expand'] = cumsum / cumcnt

all_data['item_target_enc_expand'].fillna(0.3343, inplace = True)
encoded_feature = all_data['item_target_enc_expand'].values