import pandas as pd
import shelve
import re

def clean_column_label(label) :
  label = re.sub(r'BodyBody', 'Body', label)
  label = re.sub(r'\(\)', '', label)
  label = re.sub(r'(\(|\))', '.', label)
  label = re.sub(r'(-|_| |,)', '.', label)
  label = re.sub(r'\.\.', '.', label)
  return label

def should_drop_col(col) :
  r = False
  r = r or re.match( r'^Unnamed:.0$', col)
  r = r or re.match( r'.*\.(X|Y|Z)$', col)
  r = r or re.match( r'.*\.(min|max|mad|energy|sma|iqr|entropy|arCoeff).*', col)
  r = r or re.match( r'.*\.(bandsEnergy).*', col)
  return r

def clean_column_label_again(label) :
  label = re.sub(r'Body', '', label)
  label = re.sub(r'Mag', '', label)
  label = re.sub(r'.mean', 'Mean', label)
  label = re.sub(r'.std', 'SD', label)
  label = re.sub(r'.skewness', 'Skewness', label)
  label = re.sub(r'.kurtosis', 'Kurtosis', label)
  label = re.sub(r'.gravity', 'Gravity', label)
  label = re.sub(r'angle.t?', 'Angle', label)
  return label

def get_clean_data() :
  df = pd.read_csv('../datasets/samsung/samsungdata.csv')
  df.columns = [clean_column_label(col) for col in df.columns]
  df.columns = df.columns.drop_duplicates()
  cols_to_drop = [col for col in df.columns if should_drop_col(col)]
  df = df.drop(cols_to_drop, axis=1)
  df.columns = [clean_column_label_again(col) for col in df.columns]
  return df

s = shelve.open('random-forests-shelf.db')
try:
  if not 'df' in s : s['df'] = get_clean_data()
  df = s['df']

  samtrain = df[df['subject'] <= 12]
  samval = df[df['subject'] > 12][df['subject'] <= 17]
  samtest = df[df['subject'] > 17]


  # We use the Python RandomForest package from the scikits.learn collection of algorithms.
  # The package is called sklearn.ensemble.RandomForestClassifier

  # For this we need to convert the target column ('activity') to integer values
  # because the Python RandomForest package requires that.
  # In R it would have been a "factor" type and R would have used that for classification.

  # We map activity to an integer according to
  # laying = 1, sitting = 2, standing = 3, walk = 4, walkup = 5, walkdown = 6
  # Code is in supporting library randomforest.py

  import randomforests as rf
  samtrain = rf.remap_col(samtrain,'activity')
  samval = rf.remap_col(samval,'activity')
  samtest = rf.remap_col(samtest,'activity')

  import sklearn.ensemble as sk
  #rfc = sk.RandomForestClassifier(n_estimators=500, compute_importances=True, oob_score=True)
  rfc = sk.RandomForestClassifier(n_estimators=500, oob_score=True)

  train_data = samtrain[samtrain.columns[1:-2]]
  train_truth = samtrain['activity']
  model = rfc.fit(train_data, train_truth)

  # use the OOB (out of band) score which is an estimate of accuracy of our model.
  print rfc.oob_score_

  ### TRY THIS
  # use "feature importance" scores to see what the top 10 important features are
  fi = enumerate(rfc.feature_importances_)
  cols = samtrain.columns
  print [(value,cols[i]) for (i,value) in fi if value > 0.04]
  ## Change the value 0.04 which we picked empirically to give us 10 variables
  ## try running this code after changing the value up and down so you get more or less variables
  ## do you see how this might be useful in refining the model?
  ## Here is the code in case you mess up the line above
  ## [(value,cols[i]) for (i,value) in fi if value > 0.04]

  # pandas data frame adds a spurious unknown column in 0 position hence starting at col 1
  # not using subject column, activity ie target is in last columns hence -2 i.e dropping last 2 cols

  val_data = samval[samval.columns[1:-2]]
  val_truth = samval['activity']
  val_pred = rfc.predict(val_data)

  test_data = samtest[samtest.columns[1:-2]]
  test_truth = samtest['activity']
  test_pred = rfc.predict(test_data)

  print("mean accuracy score for validation set = %f" %(rfc.score(val_data, val_truth)))
  print("mean accuracy score for test set = %f" %(rfc.score(test_data, test_truth)))

finally:
    s.close()




