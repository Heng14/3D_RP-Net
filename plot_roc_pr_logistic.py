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
    preds0 = []
    preds1 = []
    y_test = []
    with open(input_txt,'r') as fin:
        for line in fin:
            line = line.strip().split(' ')
            preds0.append(float(line[0]))
            preds1.append(float(line[1]))
            y_test.append(float(line[-1]))

    fpr0, tpr0, thresh_roc0 = roc_curve(y_test,preds0)
    fpr1, tpr1, thresh_roc1 = roc_curve(y_test,preds1)

    pre0, rec0, thresh_pr0 = precision_recall_curve(y_test,preds0)
    pre1, rec1, thresh_pr1 = precision_recall_curve(y_test,preds1)

    return fpr0, tpr0, thresh_roc0, fpr1, tpr1, thresh_roc1, pre0, rec0, thresh_pr0, pre1, rec1, thresh_pr1, preds0, preds1, y_test

def process_one(f_p, f_o, f, mode, thres0=None, thres0_acc=None, thres1=None, thres1_acc=None):

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

    #train_txt = osp.join(f_p, 'train_log.txt')

    fpr0, tpr0, thresh_roc0, fpr1, tpr1, thresh_roc1, pre0, rec0, thresh_pr0, pre1, rec1, thresh_pr1, preds0, preds1, y_test = process_txt(f_p)

    if mode == 'roc':
        thresh_c0, acc0, sp0, sen0, thresh_accc0 = cal_roc_acc(fpr0, tpr0, thresh_roc0, preds0, y_test, thres0, thres0_acc)
        auc_num0 = auc(fpr0,tpr0)
        #plt.plot(fpr0,tpr0,color='blue',lw=lw, label=f'cnn (auc=%0.4f,th=%0.4f,acc=%0.4f,sp=%0.4f,sen=%0.4f)'%(auc_num0,thresh_c0,acc0,sp0,sen0))
        plt.plot(fpr0,tpr0,color='blue',lw=lw, label=f'cnn (auc=%0.4f,acc=%0.4f,sp=%0.4f,sen=%0.4f)'%(auc_num0,acc0,sp0,sen0))

        thresh_c1, acc1, sp1, sen1, thresh_accc1 = cal_roc_acc(fpr1, tpr1, thresh_roc1, preds1, y_test, thres1, thres1_acc)
        auc_num1 = auc(fpr1,tpr1)
        #plt.plot(fpr1,tpr1,color='red',lw=lw, label=f'c_c (auc=%0.4f,th=%0.4f,acc=%0.4f,sp=%0.4f,sen=%0.4f)'%(auc_num1,thresh_c1,acc1,sp1,sen1))
        plt.plot(fpr1,tpr1,color='red',lw=lw, label=f'c_T (auc=%0.4f,acc=%0.4f,sp=%0.4f,sen=%0.4f)'%(auc_num1,acc1,sp1,sen1))

    elif mode == 'pr':
        auc_num0 = auc(rec0, pre0)
        plt.plot(rec0, pre0,color='blue',lw=lw, label=f'cnn (auc=%0.4f)'%auc_num0)
        auc_num1 = auc(rec1, pre1)
        plt.plot(rec1, pre1,color='red',lw=lw, label=f'c_T (auc=%0.4f)'%auc_num1)

    '''
    val_txt = osp.join(f_p, 'val_log.txt')
    fpr, tpr, thresh_roc, pre, rec, thresh_pr = process_txt(val_txt)
    if mode == 'roc':
        auc_num = auc(fpr,tpr)
        plt.plot(fpr,tpr,color='blue',lw=lw, label=f'val curve (area=%0.4f)'%auc_num)
    elif mode == 'pr':
        auc_num = auc(rec, pre)
        plt.plot(rec, pre,color='blue',lw=lw, label=f'val curve (area=%0.4f)'%auc_num)

    test_txt = osp.join(f_p, 'test_log.txt')
    fpr, tpr, thresh_roc, pre, rec, thresh_pr = process_txt(test_txt)
    if mode == 'roc':
        auc_num = auc(fpr,tpr)
        plt.plot(fpr,tpr,color='green',lw=lw, label=f'test curve (area=%0.4f)'%auc_num)
    elif mode == 'pr':
        auc_num = auc(rec, pre)
        plt.plot(rec, pre,color='green',lw=lw, label=f'test curve (area=%0.4f)'%auc_num)
    '''

    plt.legend(loc='lower right')
    plt.savefig(osp.join(f_o, f'{f}_{mode}.png'))
    plt.close()

    if mode == 'roc':
            return thresh_c0, thresh_accc0, thresh_c1, thresh_accc1

def filename(f):
    return osp.basename(osp.splitext(f)[0])

def main(f_path):

    f_train = osp.join(f_path, 'train_T.txt')
    f_val = osp.join(f_path, 'val_T.txt')
    f_test = osp.join(f_path, 'test_T.txt')

    #os.makedirs(f_o, exist_ok=True)
    #train
    f_name = filename(f_train)
    thresh_c0, thresh_accc0, thresh_c1, thresh_accc1 = process_one(f_train, f_path, f_name, 'roc')
    #process_one(f_train, f_path, f_name, 'pr')

    #val
    f_name = filename(f_val)
    process_one(f_val, f_path, f_name, 'roc', thresh_c0, thresh_accc0, thresh_c1, thresh_accc1)
    #process_one(f_val, f_path, f_name, 'pr')

    #test
    f_name = filename(f_test)
    process_one(f_test, f_path, f_name, 'roc', thresh_c0, thresh_accc0, thresh_c1, thresh_accc1)
    #process_one(f_test, f_path, f_name, 'pr')
    return

def process_three(f_path):
    f_train = osp.join(f_path, 'train_T.txt')
    f_val = osp.join(f_path, 'val_T.txt')
    f_test = osp.join(f_path, 'test_T.txt')
    
    #mode = 'roc'
    mode = 'pr'

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


    thres0=None; thres0_acc=None;
    fpr0, tpr0, thresh_roc0, fpr1, tpr1, thresh_roc1, pre0, rec0, thresh_pr0, pre1, rec1, thresh_pr1, preds0, preds1, y_test = process_txt(f_train)
    if mode == 'roc':
        thresh_c0, acc0, sp0, sen0, thresh_accc0 = cal_roc_acc(fpr0, tpr0, thresh_roc0, preds0, y_test, thres0, thres0_acc)
        auc_num0 = auc(fpr0,tpr0)
        #plt.plot(fpr0,tpr0,color='blue',lw=lw, label=f'cnn (auc=%0.4f,th=%0.4f,acc=%0.4f,sp=%0.4f,sen=%0.4f)'%(auc_num0,thresh_c0,acc0,sp0,sen0))
        plt.plot(fpr0,tpr0,color='blue',lw=lw, label=f'Train (auc=%0.3f,sp=%0.3f,sen=%0.3f)'%(auc_num0,sp0,sen0))

    elif mode == 'pr':
        auc_num0 = auc(rec0, pre0)
        #plt.plot(rec0, pre0,color='blue',lw=lw, label=f'cnn (auc=%0.4f)'%auc_num0)
        plt.plot(rec0, pre0,color='blue',lw=lw, label=f'Train (auc=%0.4f)'%auc_num0)

    fpr0, tpr0, thresh_roc0, fpr1, tpr1, thresh_roc1, pre0, rec0, thresh_pr0, pre1, rec1, thresh_pr1, preds0, preds1, y_test = process_txt(f_val)
    if mode == 'roc':
        thres0=thresh_c0; thres0_acc=thresh_accc0;
        thresh_c0, acc0, sp0, sen0, thresh_accc0 = cal_roc_acc(fpr0, tpr0, thresh_roc0, preds0, y_test, thres0, thres0_acc)
        auc_num0 = auc(fpr0,tpr0)
        #plt.plot(fpr0,tpr0,color='blue',lw=lw, label=f'cnn (auc=%0.4f,th=%0.4f,acc=%0.4f,sp=%0.4f,sen=%0.4f)'%(auc_num0,thresh_c0,acc0,sp0,sen0))
        plt.plot(fpr0,tpr0,color='green',lw=lw, label=f'Int_val (auc=%0.3f,sp=%0.3f,sen=%0.3f)'%(auc_num0,sp0,sen0))
    elif mode == 'pr':
        auc_num0 = auc(rec0, pre0)
        #plt.plot(rec0, pre0,color='blue',lw=lw, label=f'cnn (auc=%0.4f)'%auc_num0)
        plt.plot(rec0, pre0,color='green',lw=lw, label=f'Int_val (auc=%0.4f)'%auc_num0)

    fpr0, tpr0, thresh_roc0, fpr1, tpr1, thresh_roc1, pre0, rec0, thresh_pr0, pre1, rec1, thresh_pr1, preds0, preds1, y_test = process_txt(f_test)
    if mode == 'roc':
        thres0=thresh_c0; thres0_acc=thresh_accc0;
        thresh_c0, acc0, sp0, sen0, thresh_accc0 = cal_roc_acc(fpr0, tpr0, thresh_roc0, preds0, y_test, thres0, thres0_acc)
        auc_num0 = auc(fpr0,tpr0)
        #plt.plot(fpr0,tpr0,color='blue',lw=lw, label=f'cnn (auc=%0.4f,th=%0.4f,acc=%0.4f,sp=%0.4f,sen=%0.4f)'%(auc_num0,thresh_c0,acc0,sp0,sen0))
        plt.plot(fpr0,tpr0,color='red',lw=lw, label=f'Ext_val (auc=%0.3f,sp=%0.3f,sen=%0.3f)'%(auc_num0,sp0,sen0))
    elif mode == 'pr':
        auc_num0 = auc(rec0, pre0)
        #plt.plot(rec0, pre0,color='blue',lw=lw, label=f'cnn (auc=%0.4f)'%auc_num0)
        plt.plot(rec0, pre0,color='red',lw=lw, label=f'Ext_val (auc=%0.4f)'%auc_num0)


    plt.legend(loc='lower right')
    plt.savefig(osp.join(f_path, f'all_{mode}.png'))
    plt.close()

if __name__=='__main__':
    import fire
    #fire.Fire(main)
    fire.Fire(process_three)



