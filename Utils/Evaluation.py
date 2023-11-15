import medpy 
from medpy.metric.binary import dc, jc, asd, assd, precision, recall, sensitivity, specificity 

def multiclass_metrics(true_masks, 
                      predicted_masks, 
                      n_class, 
                      metrics=[dc, jc, asd, assd, precision, recall, sensitivity, specificity]):

  all_results = []
  for metric in metrics:

    tot_val = [0] * len(predicted_masks)
    avg_val = [0] * n_class

    for cl in range(n_class):
      for pred, tru in zip(predicted_masks, true_masks):
        tot_val[cl] += metric(pred==cl, tru==cl)
      avg_val[cl] = tot_val[cl]/float(len(predicted_masks))
    all_results.append((metric.__name__, avg_val))

  return all_results