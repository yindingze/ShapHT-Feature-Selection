from sklearn.ensemble import RandomForestClassifier
from Feature_Selection import ShapHT
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# %%
# 生成分类特征集

sample_num = 1000 # 样本数
feature_num = 100 # 特征数
class_num = 5 # 分类任务的类别数
relevant_num = 20 # 相关特征数

feature, label = make_classification(n_samples=sample_num, n_features=feature_num, n_informative=relevant_num,
                           n_redundant=0, n_repeated=0, n_classes=class_num,
                           n_clusters_per_class=1, weights=None,
                           flip_y=0.01, class_sep=1.0, hypercube=True,
                           shift=0.0, scale=1.0, shuffle=False, random_state=0)


# %%
train_feature, test_feature, train_label, test_label = train_test_split(feature, label, test_size=0.3, random_state=0)

# %%
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(train_feature, train_label)
score = clf.score(test_feature,test_label)
print("Before Feature Selection ACC: "+str(score))
# %%
# ShapHT+ 特征选择算法
params = clf.get_params()
feat_selector = ShapHT.ShapHT(clf, params)
feat_selector.fit(train_feature, train_label)

# 相关特征
support = feat_selector.support_s + feat_selector.support_w

#%%
select_train_feature = feat_selector.transform(train_feature)
select_test_feature = feat_selector.transform(test_feature)
#%%
clf.fit(select_train_feature, train_label)
score = clf.score(select_test_feature,test_label)
print("After Feature Selection ACC: "+str(score))

