rows = test.FirstId.append(test.SecondId)

cols = test.SecondId.append(test.FirstId)

rows_cols = pd.concat([rows, cols], axis=1)

rows_cols = rows_cols.drop_duplicates()

data = np.ones(rows_cols.shape[0], dtype=int)

inc_mat = scipy.sparse.coo_matrix((data, (rows_cols[0], rows_cols[1])), dtype=int)

inc_mat = inc_mat.tocsr()

## Cosine distance/feature vector
f = rows_FirstId.multiply(rows_SecondId) 
f = f.sum(axis=1)

f = np.squeeze(np.asarray(f))