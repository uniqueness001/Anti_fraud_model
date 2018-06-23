# coding:utf-8
# @author:zee(GDUT)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
#  忽略弹出的warnings
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('D:/Anti_fraud_model/creditcard.csv',encoding='latin-1')
def processing():
        sns.set_style('whitegrid')
        import missingno as msno
        # data = np.loadtxt(open("D:/Anti_fraud_model/creditcard.csv", "rb"), delimiter=",", skiprows=0)
        # print(data.head(5))
        # 数据集为结构化数据，不需要抽特征转化，但特征Time和Amount的数据规格和其他特征不一样，需要对其做特征做特征缩放。
        # 通过查看数据信息得知，数据的类型基本是float64和int64数据类型。
        # print(data.info())
        # 查看数据基本统计信息
        # print(data.describe().T)
        # 查看缺失值情况
        # print(msno.matrix(data))
        # 查看目标列的情况
        # print(data.groupby('Class').size())
        # %matplotlib inline
        # 目标变量分布可视化
        sns.set_style('whitegrid')
        fig, axs = plt.subplots(1,2,figsize=(14,7))
        sns.countplot(x='Class',data=data,ax=axs[0])
        axs[0].set_title("Frequency of each Class")
        data['Class'].value_counts().plot(x=None,y=None, kind='pie', ax=axs[1],autopct='%1.2f%%')
        axs[1].set_title("Percentage of each Class")
        # plt.show()
        # 特征Time的单为秒，我们将其转化为以小时为单位对应每天的时间
        data['Hour'] =data["Time"].apply(lambda x : divmod(x, 3600)[0])
        # 看下信用卡消费的时间
        sns.factorplot(x="Hour", data=data, kind="count",  palette="ocean", size=6, aspect=3)
        # 盗刷交易、交易金额和交易时间的关系
        # f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16,6))
        # ax1.scatter(data["Hour"][data["Class"] == 1], data["Amount"][data["Class"]  == 1])
        # ax1.set_title('Fraud')
        # ax2.scatter(data["Hour"][data["Class"] == 0], data["Amount"][data["Class"] == 0])
        # ax2.set_title('Normal')
        # plt.xlabel('Time (in Hours)')
        # plt.ylabel('Amount')
        # plt.show()
        # 输出具体信息
        # print ("Fraud Stats Summary")
        # print (data["Amount"][data["Class"] == 1].describe())
        # print ()
        # print ("Normal Stats Summary")
        # print (data["Amount"][data["Class"]  == 0].describe())
        # 不同变量在信用卡被盗刷和信用卡正常的不同分布情况，我们将选择在不同信用卡状态下的分布有明显区别的变量
        v_feat = data.ix[:, 1:29].columns
        plt.figure(figsize=(16, 28*4))
        gs = gridspec.GridSpec(28, 1)
        for i, cn in enumerate(data[v_feat]):
            ax = plt.subplot(gs[i])
            sns.distplot(data[cn][data["Class"] == 1], bins=50)
            sns.distplot(data[cn][data["Class"] == 0], bins=100)
            ax.set_xlabel('')
            ax.set_title('histogram of feature: ' + str(cn))
            # 因此剔除变量V8、V13 、V15 、V20 、V21 、V22、 V23 、V24 、V25 、V26 、V27 和V28变量。这也与我们开始用相关性图谱观察得出结论一致。同时剔除变量Time，保留离散程度更小的Hour变量。
        droplist = ['V8', 'V13', 'V15', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28','Time']
        data_new = data.drop(droplist, axis = 1)
        # print(data_new.shape) # 查看数据的维度
        # 对Amount和Hour 进行标准化处理
        col = ['Amount', 'Hour']
        sc =StandardScaler() # 初始化缩放器
        data_new[col] =sc.fit_transform(data_new[col])#对数据进行标准化
        # 变量之间相关性分析
        # data.corr() #相关系数矩阵，即给出了任意两款菜式之间的相关系数
        # data.corr()[u'Hour'] #只显示“Hour”与其他变量的相关系数
        #data[u'Hour'].corr(data[u'Time']) #计算“Hour”与“Time”的相关系数
        return(data_new)
def feature_importance():
        x_feature = list(data.columns)
        x_feature.remove('Class')
        x_val = data[x_feature]
        y_val = data['Class']
        names = data[x_feature].columns
        clf = RandomForestClassifier(n_estimators=10, random_state=123)  # 构建分类随机森林分类器
        clf.fit(x_val, y_val)  # 对自变量和因变量进行拟合
        # names, clf.feature_importances_
        for feature in zip(names, clf.feature_importances_):
            return(feature)
if __name__=='__main__':
    result = processing()
    # print(result.info())
    result01 = feature_importance()
    print(result01)