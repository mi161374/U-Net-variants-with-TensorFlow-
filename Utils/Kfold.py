from sklearn.model_selection import train_test_split, KFold

def kfold_split(images, 
                masks, 
                train_test_split_ratio=0.11, 
                split_shuffle=True, 
                k_val=10, 
                k_shuffle=True, 
                K_random_state=1):

  x_train_kf = []
  x_val_kf = []
  x_test_kf = []
  y_train_kf = []
  y_val_kf = []
  y_test_kf = []

  kf = KFold(n_splits=k_val, shuffle=k_shuffle, random_state=K_random_state)

  for train_index, test_index in kf.split(images):
    x_1, x_2, y_1, y_2 = train_test_split(images[train_index], 
                                          masks[train_index], 
                                          test_size = train_test_split_ratio, 
                                          shuffle=split_shuffle)
    x_train_kf.append(x_1)
    x_val_kf.append(x_2)
    x_test_kf.append(images[test_index])
    y_train_kf.append(y_1)
    y_val_kf.append(y_2)
    y_test_kf.append(masks[test_index])
  return x_train_kf, x_val_kf, x_test_kf, y_train_kf, y_val_kf, y_test_kf