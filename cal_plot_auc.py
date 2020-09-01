import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def cal_dis(x, y):
    return x**2+(1-y)**2

def cal_acc(thresh, preds, y_test):
    preds = np.array(preds)
    y_test = np.array(y_test)
    return len(np.where((preds>thresh)==y_test)[0])/len(y_test)


def cal_PN(thres, preds, y_test):
    preds = np.array(preds)
    y_test = np.array(y_test)
    tp = len((np.where((preds>=thres) & (y_test>0)))[0])
    fp = len((np.where((preds>=thres) & (y_test<1)))[0])
    fn = len((np.where((preds<thres) & (y_test>0)))[0])
    tn = len((np.where((preds<thres) & (y_test<1)))[0])

    assert tp+fp+fn+tn==len(y_test), print ('tp+fp+fn+tn not equal to y_test !!!')
    fpr = fp/(fp+tn)
    tpr = tp/(tp+fn)
    return fpr, tpr

def cal_roc_acc(fpr, tpr, thresh, preds, y_test, thres, thres_acc):
    if thres:
        print('thresh: ', thres)
        acc = cal_acc(thres_acc, preds, y_test)
        print ('acc: ', acc)
        fpr_c, tpr_c = cal_PN(thres, preds, y_test)
        print('fpr: ',fpr_c)
        print ('tpr: ',tpr_c)
        return thres, acc, 1-fpr_c, tpr_c, thres_acc
    assert len(fpr)==len(tpr)==len(thresh), print ('fpr tpr thresh len not equal !!!')


    '''
    accs = [cal_acc(i, preds, y_test) for i in thresh]
    acc = max(accs)
    thresh_c = thresh[accs.index(acc)]
    fpr_c = fpr[accs.index(acc)]
    tpr_c = tpr[accs.index(acc)]
    print('thresh: ',thresh_c)
    print ('acc: ',acc)
    print('fpr: ',fpr_c)
    print ('tpr: ',tpr_c)
    return thresh_c, acc, 1-fpr_c, tpr_c
    '''

    accs = [cal_acc(i, preds, y_test) for i in thresh]
    acc_c = max(accs)
    thresh_acc = thresh[accs.index(acc_c)]

    dis = [cal_dis(fpr[i], tpr[i]) for i in range(len(thresh))]
    dis_c = min(dis)
    thresh_c = thresh[dis.index(dis_c)]
    acc = cal_acc(thresh_c, preds, y_test)
    fpr_c = fpr[dis.index(dis_c)]
    tpr_c = tpr[dis.index(dis_c)]
    print('thresh: ',thresh_c)
    print ('acc:' ,cal_acc(thresh_c, preds, y_test))
    print('fpr: ',fpr_c)
    print ('tpr: ',tpr_c)
    return thresh_c, acc_c, 1-fpr_c, tpr_c, thresh_acc
    

def process_txt(input_txt):
    preds = []
    y_test = []
    with open(input_txt,'r') as fin:
        for line in fin:
            line = line.strip().split(' ')
            preds.append(float(line[1]))
            y_test.append(float(line[3]))

    fpr, tpr, thresh_roc = roc_curve(y_test,preds)
    pre, rec, thresh_pr = precision_recall_curve(y_test,preds)
    return fpr, tpr, thresh_roc, pre, rec, thresh_pr, preds, y_test

def process_one(f_p, f_o, f, mode):

    plt.figure()
    lw = 2
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    if mode == 'roc':
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC curve')
        plt.plot([0,1],[0,1], color='darkorange', lw=lw, linestyle='-')
    elif mode == 'pr':
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR curve')


    train_txt = osp.join(f_p, 'train_log.txt')
    fpr0, tpr0, thresh_roc0, pre, rec, thresh_pr, preds0, y_test0 = process_txt(train_txt)
    if mode == 'roc':
        auc_num0 = auc(fpr0,tpr0)
        thresh_c, acc0, sp0, sen0, thresh_acc = cal_roc_acc(fpr0, tpr0, thresh_roc0, preds0, y_test0, None, None)
        plt.plot(fpr0,tpr0,color='red',lw=lw, label=f'train (auc=%0.4f,th=%0.4f,acc=%0.4f,sp=%0.4f,sen=%0.4f)'%(auc_num0,thresh_c,acc0,sp0,sen0))
    elif mode == 'pr':
        auc_num = auc(rec, pre)
        plt.plot(rec, pre,color='red',lw=lw, label=f'train curve (area=%0.4f)'%auc_num)

    val_txt = osp.join(f_p, 'val_log.txt')
    fpr1, tpr1, thresh_roc1, pre, rec, thresh_pr, preds1, y_test1 = process_txt(val_txt)
    if mode == 'roc':
        auc_num1 = auc(fpr1,tpr1)
        _, acc1, sp1, sen1, _ = cal_roc_acc(fpr1, tpr1, thresh_roc1, preds1, y_test1, thresh_c, thresh_acc)
        plt.plot(fpr1,tpr1,color='blue',lw=lw, label=f'val (auc=%0.4f,th=%0.4f,acc=%0.4f,sp=%0.4f,sen=%0.4f)'%(auc_num1,thresh_c,acc1,sp1,sen1))
    elif mode == 'pr':
        auc_num = auc(rec, pre)
        plt.plot(rec, pre,color='blue',lw=lw, label=f'val curve (area=%0.4f)'%auc_num)

    test_txt = osp.join(f_p, 'test_log.txt')
    fpr2, tpr2, thresh_roc2, pre, rec, thresh_pr, preds2, y_test2 = process_txt(test_txt)
    if mode == 'roc':
        auc_num2 = auc(fpr2,tpr2)
        _, acc2, sp2, sen2, _ = cal_roc_acc(fpr2, tpr2, thresh_roc2, preds2, y_test2, thresh_c, thresh_acc)
        plt.plot(fpr2,tpr2,color='green',lw=lw, label=f'test (auc=%0.4f,th=%0.4f,acc=%0.4f,sp=%0.4f,sen=%0.4f)'%(auc_num2,thresh_c,acc2,sp2,sen2))
    elif mode == 'pr':
        auc_num = auc(rec, pre)
        plt.plot(rec, pre,color='green',lw=lw, label=f'test curve (area=%0.4f)'%auc_num)

    plt.legend(loc='lower right')
    plt.savefig(osp.join(f_o, f'{f}_roc.png'))
    plt.close()


def main(f_path):
    fs = os.listdir(f_path)
    out_p = 'result_plot'
    out_roc = osp.join(out_p, 'roc')
    out_pr = osp.join(out_p, 'pr')
    os.makedirs(out_roc, exist_ok=True)
    os.makedirs(out_pr, exist_ok=True)
    for f in fs:
        f_p = osp.join(f_path, f)
        process_one(f_p, out_roc, f, 'roc')
        process_one(f_p, out_pr, f, 'pr')

if __name__=='__main__':
    import fire
    fire.Fire(main)



