import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os
import numpy as np

def draw_roc(fpr,tpr,set_curve):
    path_to_file = set_curve + '.png'
    auc_num = auc(fpr,tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr,tpr,color='darkorange',lw=lw, label=f'{set_curve} curve (area=%0.2f)'%auc_num)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    if set_curve=='roc':
        plt.plot([0,1],[0,1], color='navy', lw=lw, linestyle='-')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
    elif set_curve=='pr':
        plt.xlabel('Recall')
        plt.ylabel('Precision')
    else:
        print ('set_curve error !!!')
        raise
    plt.title(f'{set_curve} curve of class-PCR')
    plt.legend(loc='lower right')
    plt.savefig(path_to_file)

def cal_dis(x, y):
    return x**2+(1-y)**2

def cal_acc(thresh, preds, y_test):
    preds = np.array(preds)
    y_test = np.array(y_test)
    return len(np.where((preds>thresh)==y_test)[0])/len(y_test)

def cal_roc_acc(fpr, tpr, thresh, preds, y_test):
    assert len(fpr)==len(tpr)==len(thresh), print ('fpr tpr thresh len not equal !!!')
    dis = [cal_dis(fpr[i], tpr[i]) for i in range(len(thresh))]
    thresh_c = thresh[dis.index(min(dis))]
    print('thresh: ',thresh_c)
    print ('acc:' ,cal_acc(thresh_c, preds, y_test))

def cal_pr_acc(thresh, preds, y_test, thres):
    if thres:
        print('thresh: ',thres)
        print ('acc:' ,cal_acc(thres, preds, y_test))
        return
    accs = [cal_acc(i, preds, y_test) for i in thresh]
    acc = max(accs)
    thresh_c = thresh[accs.index(acc)]
    print('thresh: ',thresh_c)
    print ('acc:' ,acc)

def main(input_txt, thres=None, set_curve='pr'):
    preds = []
    y_test = []
    with open(input_txt,'r') as fin:
        for line in fin:
            line = line.strip().split(' ')
            preds.append(float(line[1]))
            y_test.append(float(line[3]))

    #print(preds)
    #print(y_test)
    #preds = np.array(preds)
    #y_test = np.array(y_test)
    #print (len(np.where((preds>0.5)==y_test)[0])/len(y_test))
    if set_curve == 'roc':
        fpr, tpr, thresh = roc_curve(y_test,preds)        
        #print('FPR —————————')
        #print(fpr)
        #print('TPR —————————')
        #print(tpr)
        #print ('thresh ---------')
        #print (thresh)
        draw_roc(fpr, tpr, set_curve)
        cal_roc_acc(fpr, tpr, thresh, preds, y_test)

    elif set_curve == 'pr':
        pre, rec, thresh = precision_recall_curve(y_test,preds)
        #print('FPR —————————')
        #print(fpr)
        #print('TPR —————————')
        #print(tpr)
        #print ('thresh ---------')
        #print (thresh)
        draw_roc(rec, pre, set_curve)
        cal_pr_acc(thresh, preds, y_test, thres)

    else:
        print ('set_curve error !!!')
        raise

if __name__ == '__main__':
    import fire
    fire.Fire(main)


