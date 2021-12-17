from sklearn.ensemble import RandomForestRegressor
from Feature_Selection import ShapHT
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# %%
# 生成回归特征集

sample_num = 1000 # 样本数
feature_num = 50 # 特征数
relevant_num = 10 # 相关特征数

feature, label = make_regression(n_samples=sample_num, n_features=feature_num, n_informative=relevant_num,
                    n_targets=1, bias=0.0, effective_rank=None,
                    tail_strength=0.5, noise=0.0, shuffle=False, coef=False,
                    random_state=0)

# %%
train_feature, test_feature, train_label, test_label = train_test_split(feature, label, test_size=0.3, random_state=0)

# %%
clf = RandomForestRegressor(n_estimators=100, random_state=0)
clf.fit(train_feature, train_label)
score = clf.score(test_feature,test_label)
print("Before Feature Selection ACC: "+str(score))
# %%
# ShapHT+ 特征选择算法
params = clf.get_params()
feat_selector = ShapHT.ShapHT(clf, params, isRegression=True)
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