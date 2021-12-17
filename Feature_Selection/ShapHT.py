import numpy as np
import scipy as sp
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
import shap
import math


class ShapHT(BaseEstimator, TransformerMixin):

    def __init__(self, estimator, params, alpha=0.05, delta=0.1, max_delta=1, max_iter=100, random_s=0,
                 verbose=0, isRegression=False):
        """
        初始化
        Args:
            estimator: 特征集所使用的分类/回归随机森林的对象
            params: 随机森林的参数
            alpha: 假设检验的显著性水平 默认 0.05
            delta: 初始阈值系数 (0,1] 默认 0.1
            max_delta: 最大阈值系数 <= 1 默认 1
            max_iter: 最大迭代次数
            random_s: 随机数种子
            verbose: 过程可视化
            isRegression: 是否为回归任务 默认为分类任务（False）
        """
        self.estimator = estimator
        self.params = params
        self.alpha = alpha
        self.max_iter = max_iter
        self.verbose = verbose
        self.isRegression = isRegression
        self.random_s = random_s
        self.delta = delta
        self.max_delta = max_delta

    def transform(self, X):
        support = self.support_s + self.support_w
        return X[:,support]

    def fit(self, X, y):
        """

        Args:
            X: 特征集 维度为 [样本数,特征数]
            y: 标签

        Returns:

        """
        feanum = X.shape[1] #特征数
        self._init_fit(feanum) #函数初始化
        self._fit(X, y) #特征选择过程

    def _init_fit(self, feanum):
        self.support_s = np.zeros(feanum, dtype=np.bool) #强相关特征集合
        self.support_w = np.ones(feanum, dtype=np.bool) #弱相关特征集合 （算法过程中为待定特征集）
        self.support_ir = np.zeros(feanum, dtype=np.bool) #不相关特征集合
        self.hits = np.zeros([self.max_iter, feanum]) #相关性指标
        self.imp_k = [] #重要性指标
        self.l_imp_max = []
        self.sha_max_history = []
        self.imp_history = np.zeros(feanum, dtype=np.float)
        self.d = (self.max_delta - self.delta) / (math.sqrt(self.max_iter))

    def _fit(self, X, y):
        feanum = X.shape[1]
        break_i = 0
        delet_num = 0
        old_num = 0
        hit_iter = 0
        all_hit = np.zeros(feanum)

        for _iter in range(self.max_iter):

            self.random_s = self.random_s + _iter
            self._init_support(feanum)
            shuffle_num = self._change_shuffle_num()

            X_iter, y_iter, X_iter_i, feanum_X_iter = self._get_X_iter(X, y)
            X_shap = self._get_X_shap(X_iter, shuffle_num)
            shap_values = self._get_shap_values(X_shap, y_iter)

            DeltaTF = True

            while (DeltaTF):
                hit_iter = hit_iter + 1
                all_hit_p = np.zeros(X_shap.shape[1])
                all_hit_p[0:feanum_X_iter] = all_hit[X_iter_i]
                hit, imp_max, all_hit_p = self._hit(feanum, X_iter_i, shap_values, feanum_X_iter, self.delta, all_hit_p,
                                                    y_iter)
                self._select(X_iter_i, hit, all_hit_p, feanum_X_iter, X_iter.shape[0], hit_iter)
                all_hit[X_iter_i] = all_hit_p[0:feanum_X_iter]
                old_num = delet_num
                delet_num = np.where(self.support_ir == True)[0].shape[0]

                DeltaTF = self._check_num(old_num, delet_num, DeltaTF)

            if self.verbose == 1:
                print(" 已删除：", np.where(self.support_ir == True)[0].shape[0])

            self.hits[_iter, X_iter_i] = hit[0:feanum_X_iter]
            self.imp_k.append(np.fabs(shap_values[:, 0:feanum_X_iter]))
            self.l_imp_max.append(imp_max)

            week_num = np.where(self.support_w == True)[0].shape[0]

            TF, break_i = self._check_delta(old_num, delet_num, week_num, break_i)
            if TF == 1:
                break

        tentative = np.where(self.support_w == True)[0]
        # ignore the first row of zeros
        tentative_median = np.median(self.imp_history[1:, tentative], axis=0)
        # which tentative to keep
        tentative_confirmed = np.where(tentative_median
                                       > np.median(self.sha_max_history))[0]
        tentative = tentative[tentative_confirmed]

        self.support_weak_ = np.zeros(feanum, dtype=np.bool)
        self.support_weak_[tentative] = 1

        self.all_hit = all_hit
        return self

    def _check_delta(self, old_num, delet_num, week_num, break_i):
        TF = 0
        if self.max_delta - self.delta < 1e-6:
            if old_num != delet_num:
                break_i = 0
            elif old_num == delet_num:
                break_i = break_i + 1
            if week_num != 0:
                break_i = 0
            if break_i >= 2:
                TF = 1
        return TF, break_i

    def _check_num(self, old_num, delet_num, DeltaTF):

        if old_num != delet_num:
            DeltaTF = False
        if self.max_delta - self.delta < 1e-6:
            DeltaTF = False
        if self.max_delta - self.delta > 1e-6 and old_num == delet_num:
            self.delta = self.delta + self.d
            if self.delta > self.max_delta:
                self.delta = self.max_delta
        return DeltaTF

    def _change_shuffle_num(self):
        shuffle_num = np.where(self.support_w == True)[0].shape[0]
        return shuffle_num

    def _init_support(self, feanum):
        self.support_s = np.zeros(feanum, dtype=np.bool)
        self.support_w = np.ones(feanum, dtype=np.bool)
        self.support_w[np.where(self.support_ir == True)[0]] = 0

    def _select(self, X_iter_i, hit, all_hit, max_fea_iter, samnum, hit_iter):
        to_accept_ps = sp.stats.binom.sf(hit - 1, samnum, .5).flatten()
        to_reject_ps = sp.stats.binom.cdf(hit, samnum, .5).flatten()

        to_accept, pvals_corrected_ = self._fdrcorrection(to_accept_ps, self.alpha)
        to_reject, pvals_corrected_ = self._fdrcorrection(to_reject_ps, self.alpha)

        to_accept2 = to_accept_ps <= self.alpha / float(samnum)
        to_reject2 = to_reject_ps <= self.alpha / float(samnum)
        to_accept *= to_accept2
        to_reject *= to_reject2

        to_accept = to_accept[0:max_fea_iter]
        to_reject = to_reject[0:max_fea_iter]

        all_to_accept_ps = sp.stats.binom.sf(all_hit - 1, hit_iter, .5).flatten()
        all_to_reject_ps = sp.stats.binom.cdf(all_hit, hit_iter, .5).flatten()

        all_to_accept, pvals_corrected_ = self._fdrcorrection(all_to_accept_ps, self.alpha)
        all_to_reject, pvals_corrected_ = self._fdrcorrection(all_to_reject_ps, self.alpha)

        to_accept2 = all_to_accept_ps <= self.alpha / float(hit_iter)
        to_reject2 = all_to_reject_ps <= self.alpha / float(hit_iter)

        all_to_accept *= to_accept2
        all_to_reject *= to_reject2

        all_to_accept = all_to_accept[0:max_fea_iter]
        all_to_reject = all_to_reject[0:max_fea_iter]

        if np.where(all_to_accept == False)[0].shape[0] == 0:
            accept_i = X_iter_i[all_to_accept]
            reject_i = []
        else:
            to_acc_i = np.where(all_to_accept == True)
            to_re_i = np.where(all_to_reject == True)
            to_reject[to_acc_i] = 0
            to_accept[to_re_i] = 0

            accept_i = X_iter_i[to_accept]
            reject_i = X_iter_i[to_reject]

        self.support_s[accept_i] = 1
        self.support_ir[reject_i] = 1

        self.support_w[accept_i] = 0
        self.support_w[reject_i] = 0

    def _get_X_iter(self, X, y):

        X_split, test_feature, y_split, test_label = train_test_split(X, y, test_size=0.1, random_state=self.random_s)

        X_iter_i = np.where(self.support_ir == 0)[0]
        X_iter = X_split[:, X_iter_i]
        feanum_X_iter = X_iter.shape[1]

        return X_iter, y_split, X_iter_i, feanum_X_iter

    def _get_X_shap(self, X, shuffle_num):
        feanum = X.shape[1]

        np.random.seed(self.random_s)
        epsilon = X[:, np.random.choice(feanum, shuffle_num)].copy()

        np.random.seed(self.random_s)
        np.random.shuffle(epsilon)

        X_shap = np.hstack((X, epsilon))
        return X_shap

    def _get_shap_values(self, X, y):
        self.params['random_state'] = self.random_s

        clf = self.estimator.set_params(**self.params)

        clf.fit(X, y)
        flag = True
        while (flag):
            try:
                explainer = shap.TreeExplainer(clf, data=None, feature_perturbation="tree_path_dependent")
                shap_values = explainer.shap_values(X, approximate=True)
            except Exception:
                continue
            flag = False

        if self.isRegression == 1:
            return shap_values
        else:
            X_size = X.shape
            samnum = X_size[0]
            feanum = X_size[1]
            result = np.zeros((samnum, feanum), dtype=np.float)
            for i in range(samnum):
                result[i] = shap_values[y[i]][i, :]
            return result

    def _hit(self, feanum0, X_iter_i, shap_values, max_fea_iter, delta, all_hit, y_iter):

        imp_k = np.fabs(shap_values)
        imp = np.mean(imp_k, axis=0)
        imp_max_all = imp[max_fea_iter:].max()
        imp_max = imp_max_all * delta

        im_size = imp_k.shape
        feanum = im_size[1]
        samnum = im_size[0]
        hits = np.zeros(feanum)

        for i in range(samnum):
            h_i = np.where(imp_k[i, :] > imp_max)[0]
            hits[h_i] = hits[h_i] + 1

        cur_imp = np.zeros(feanum0)
        cur_imp[X_iter_i] = imp[0:max_fea_iter]
        self.imp_history = np.vstack((self.imp_history, cur_imp))

        imp_max_all = imp[max_fea_iter:].max()
        self.sha_max_history.append(imp_max_all)

        all_hit_i = np.where(imp > imp_max_all)[0]
        all_hit[all_hit_i] = all_hit[all_hit_i] + 1
        return hits, imp_max, all_hit

    def _fdrcorrection(self, pvals, alpha):
        """
        Benjamini/Hochberg p-value correction for false discovery rate, from
        statsmodels package. Included here for decoupling dependency on statsmodels.
        Parameters
        ----------
        pvals : array_like
            set of p-values of the individual tests.
        alpha : float
            error rate
        Returns
        -------
        rejected : array, bool
            True if a hypothesis is rejected, False if not
        pvalue-corrected : array
            pvalues adjusted for multiple hypothesis testing to limit FDR
        """
        pvals = np.asarray(pvals)
        pvals_sortind = np.argsort(pvals)
        pvals_sorted = np.take(pvals, pvals_sortind)
        nobs = len(pvals_sorted)
        ecdffactor = np.arange(1, nobs + 1) / float(nobs)

        reject = pvals_sorted <= ecdffactor * alpha

        if reject.any():
            rejectmax = max(np.nonzero(reject)[0])
            reject[:rejectmax] = True

        pvals_corrected_raw = pvals_sorted / ecdffactor
        pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
        pvals_corrected[pvals_corrected > 1] = 1
        # reorder p-values and rejection mask to original order of pvals
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[pvals_sortind] = reject
        return reject_, pvals_corrected_
