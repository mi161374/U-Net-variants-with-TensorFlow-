import os
import numpy as np 
from medpy.metric.binary import (dc, jc, asd, assd, precision, 
                                recall, sensitivity, specificity)
from tensorflow.keras.models import load_model

from Utils.DatasetManagement import  save_images
from Utils.Evaluation import multiclass_metrics


def test_model(x_test,
                y_test,
                model,
                save_pred=False,
                save_pred_dir='',
                file_names=[],
                n_labels=6,
                metrics=[dc, jc, asd, assd, 
                        precision, recall, sensitivity, specificity],
                save_results=False,
                save_results_dir='',
                use_thresholding = True,
                thresholds = [],
                kfold_num=0,
                verbose=0):

    # Load the model if a path is given
    if isinstance(model, str):
        assert os.path.exists(model), 'model must be a valid Path or an object'
        model = load_model(model)

    if use_thresholding:
        assert len(thresholds) == n_labels, 'one threshold must be provided for each label'


    # Predict and save images
    y_pred = model.predict(x_test, verbose=verbose)

    
    if use_thresholding:
        for i in range(n_labels):
            y_pred[:,:,:,i] = y_pred[:,:,:,i] > thresholds[i]

    y_pred_argmax = np.argmax(y_pred, axis=3)
    y_test_argmax = np.argmax(y_test, axis=3)


    if save_pred:
        save_pred_folder = os.path.join(save_pred_dir, 'Predictions_' 
                                        + model.name + '_' + str(kfold_num+1))
        if not os.path.exists(save_pred_folder):
            os.makedirs(save_pred_folder)
        save_images(y_pred_argmax, save_pred_folder, file_names)

    # Evaluate the model and save results 
    results = multiclass_metrics(y_test_argmax, y_pred_argmax, n_labels, metrics)

    if save_results:
        save_results_folder = os.path.join(save_results_dir, 'Elvaluation_results')
        if not os.path.exists(save_results_folder):
            os.makedirs(save_results_folder)

        file_name = model.name + '_' + str(kfold_num+1) + '.csv'
        save_results_path = os.path.join(save_results_folder, file_name)

        with open(save_results_path, 'w') as f:
            for item in results:
                f.write("%s,%s\n"%(item[0],', '.join(map(str, item[1]))))

    print(model.name + '_' + str(kfold_num+1) + ' tested successfully')

    return results, y_pred