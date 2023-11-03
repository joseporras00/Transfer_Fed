
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# function that generatesconfussion matrix and metrics of the classification models
def model_evaluation(label_test, label_pred):
   # Confussion matrix
    print('Confusion Matrix: \n',confusion_matrix(label_test,label_pred))
    print('\n')
    print('Confusion Matrix Norm: \n',confusion_matrix(label_test,label_pred, normalize='true'))
    print('\n')
    # AUC of the model
    auc=metrics.roc_auc_score(label_test, label_pred)
    print('ROC AUC: {}'.format(auc))
    print('-'*60)
    return auc
   
