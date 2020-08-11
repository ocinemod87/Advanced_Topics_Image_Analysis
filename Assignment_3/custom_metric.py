import numpy as np
from keras.applications.vgg16 import preprocess_input

def predict(img_dir):
    img = load_img(img_dir)
    img = np.resize(img, (224,224,3))
    img = preprocess_input(img_to_array(img))
    img = (img/255.0)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    idx = np.argmax(prediction)
    return idx
    pred = test_label_df.at[idx,'keras_index']

def second_metric(test_df):
    x = 0
    P = 0
    users = test_df['user'].value_counts()

    for index, row in users.items():
      N = 0
      su_df = test[test['user']==index]
      labels_idx = su_df['label'].value_counts()
      for index2, row2 in labels_idx.items():
        r = 0
        s_obs =  su_df[su_df['label']==index2]
        for index3, row3 in s_obs.iterrows():
          img_dir = row3['id']
          pred = test_label_df.at[idx,'keras_index']
          if pred==row3['label']:
            r += 1/row3['rank']
          temp_N = row3['label']
        if r == 0:continue
        N += r/len(s_obs.index)

      P += (N/su_df['label'].nunique())
      print(str(P)+ ' P')
      print(su_df['label'].nunique())

    U = test['user'].nunique()

    return P/U

def first_metric(test_df):
  P = 0
  users = test_df['user'].value_counts()

  for index, row in users.items():
    r = 0
    temp = test_df[test['user']==index]

    for index2, row2 in temp.iterrows():
      img_dir = row2['id']
      img = load_img(img_dir)
      img = np.resize(img, (224,224,3))
      img = preprocess_input(img_to_array(img))
      img = (img/255.0)
      img = np.expand_dims(img, axis=0)
      prediction = model.predict(img)
      idx = np.argmax(prediction)
      pred = test_label_df.at[idx,'keras_index']
      print(pred==row2['label'])
      if pred==row2['label']:
        r += 1/row2['rank']
    if r == 0:continue
    sum_r = 1/r
    temp_P = (sum_r/temp['label'].nunique())
    P += temp_P
  U = test_df['user'].nunique()

  return P/U
