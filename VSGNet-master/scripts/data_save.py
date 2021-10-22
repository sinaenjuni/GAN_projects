import json
import openpyxl
import pandas as pd


def scoreSave(loss_train,loss_val,loss_test,score_train,score_val,score_test,m_train,m_val,m_test):

    for i in range(len(loss_train)):
        total = []
        total.append(i)
        total.append(loss_train[i])
        total.append(loss_val[i])
        total.append(loss_test[i])
        total.append(score_train[i])
        total.append(score_val[i])
        total.append(score_test[i])
        total.append(m_train[i])
        total.append(m_val[i])
        total.append(m_test[i])

    return total

def to_xls(total_save) :
    wb = openpyxl.Workbook()
    sheet1 = wb['Sheet']
    sheet1.title = 'VCOCO80_lr4_prior'

    sheet1.append(
        ['epoch', 'loss_train', 'loss_val', 'loss_test', 'score_train', 'score_val', 'score_test', 'm_train', 'm_val',
         'm_test'])
    for result in total_save:
        #print(result)
        sheet1.append(result)

    wb.save('/home/aryoung/documents/train_result/VCOCO80_lr4_prior.xlsx')