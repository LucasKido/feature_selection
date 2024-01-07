import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, mutual_info_classif, SelectFromModel
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import  RandomForestClassifier
from lightgbm import LGBMClassifier
import warnings

class Classification_Evaluator:
    
    def sample_dataset(self, sample_size):
        """ Sample the dataset for wrapper methods keeping the same proportion in target
        """

        sample = self.X_train.copy()
        sample['target'] = self.y_train.copy()

        sample = (
            pd
            .concat(
                [
                    sample[sample['target']==0]
                    .sample(frac=sample_size, random_state=79),
                    sample[sample['target']==1]
                    .sample(frac=sample_size, random_state=79)
                ]
            )
            .sample(frac=1)
        )

        self.X_sample = sample.drop('target', axis=1)
        self.y_sample = sample['target']

    def information_filter(self):
        """ Apply Information Value and filter out features that the value is more than zero"""

        importances = mutual_info_classif(self.X_train, self.y_train)
        # Format a dataframe with features and their information values
        importances = pd.DataFrame({'features':self.X_train.columns, 'info_value':importances})
        # Filtering out features that the values are more than the threshold
        threshold = importances[importances['features']=='dummy']['info_value'].values[0]
        self.info_features = importances[importances['info_value']>threshold]

    def lasso_embedded(self):
        """ Apply L1 Regularization to filter out the best set of features."""

        lr = LogisticRegression(C=1, penalty='l1', solver='liblinear', random_state=79)
        lasso = SelectFromModel(lr).fit(self.X_train, self.y_train)
        importances = (
            pd.DataFrame(
                {
                    'features':self.X_train.columns,
                    'lasso':lasso.estimator_.coef_.reshape(-1)
                }
            )
        )
        self.lasso_features = (
            importances[importances['features'].isin(lasso.get_feature_names_out())]
            .reset_index(drop=True)
        )

    def rf_embedded(self):
        """ Uses the Feature Importance from a Random Forest to filter out some low importance features."""

        rf = RandomForestClassifier(n_estimators=100, random_state=79)
        rf.fit(self.X_train, self.y_train)
        importances = (
            pd.DataFrame(
                {
                    'features':self.X_train.columns, 
                    'rf':rf.feature_importances_
                }
            )
        )
        threshold = importances[importances['features']=='dummy']['rf'].values[0]
        self.rf_features = (
            importances[importances['rf']>threshold]
        )

    def bfs_wrapper(self, model, model_name):
        """ Use Backward Feature Selection to select the best set of fetures.

        Returns:
            feature_names (pd.DataFrame): Names of the best set of features.
        """

        bfs = SequentialFeatureSelector(model, k_features='best', forward=False, n_jobs=-1)
        bfs.fit(self.X_sample, self.y_sample)

        return pd.DataFrame({'features': bfs.k_feature_names_, model_name: np.ones(len(bfs.k_feature_names_))})
    
    def ffs_wrapper(self, model, model_name):
        """ Use Forward Feature Selection to select the best set of fetures.

        Returns:
            feature_names (pd.DataFrame): Names of the best set of features.
        """

        bfs = SequentialFeatureSelector(model, k_features='best', forward=True, n_jobs=-1)
        bfs.fit(self.X_sample, self.y_sample)

        return pd.DataFrame({'features': bfs.k_feature_names_, model_name: np.ones(len(bfs.k_feature_names_))})
    
    def compute_bfs_wrapper(self):
        """ Run Backward Feature Selection for Logistic Regression, Naive Bayes, Decision Tree, Random Forest
            and LGBM to check which model select each feature.

            Then select the feature that were chosen above the threshold.    
        """
        lr = LogisticRegression(max_iter=1000, random_state=79)
        nb = GaussianNB()
        dt = DecisionTreeClassifier()
        rf = RandomForestClassifier(n_estimators=10, random_state=79)
        lgbm = LGBMClassifier(n_estimators=10, random_state=79, verbose=-1)

        wrapper_summary = pd.DataFrame({'features': self.X_train.columns})
        models = [lr, rf, lgbm]
        names = ['lr', 'rf', 'lgbm']

        for model in zip(models, names):
            temp = self.bfs_wrapper(model[0], model[1])
            wrapper_summary = (
                wrapper_summary
                .merge(
                    temp,
                    how='left',
                    on='features'
                )
            )
        wrapper_summary['bfs'] = wrapper_summary.set_index('features').sum(axis=1).values

        self.bfs_features = (
            wrapper_summary[wrapper_summary['bfs']>=self.wrapper_threshold]
            [['features', 'bfs']]
            .reset_index(drop=True)
        )

    def compute_ffs_wrapper(self):
        """ Run Forward Feature Selection for Logistic Regression, Naive Bayes, Decision Tree, Random Forest
            and LGBM to check which model select each feature.
            
            Then select the feature that were chosen above the threshold.    
        """
        lr = LogisticRegression(max_iter=1000, random_state=79)
        rf = RandomForestClassifier(n_estimators=10, random_state=79, n_jobs=-1)
        lgbm = LGBMClassifier(n_estimators=10, random_state=79, verbose=-1, n_jobs=-1)

        wrapper_summary = pd.DataFrame({'features': self.X_train.columns})
        models = [lr, rf, lgbm]
        names = ['lr', 'rf', 'lgbm']

        for model in zip(models, names):
            temp = self.ffs_wrapper(model[0], model[1])
            wrapper_summary = (
                wrapper_summary
                .merge(
                    temp,
                    how='left',
                    on='features'
                )
            )
        wrapper_summary['ffs'] = wrapper_summary.set_index('features').sum(axis=1).values

        self.ffs_features = (
            wrapper_summary[wrapper_summary['ffs']>=self.wrapper_threshold]
            [['features', 'ffs']]
            .reset_index(drop=True)
        )

    def parse_summary(self):
        """ Aggregate every report for each method and sum the number of methods that the feature were chosen"""

        self.summary = (
            self.report
            .merge(
                self.info_features,
                how='left',
                on='features'
            )
            .merge(
                self.lasso_features,
                how='left',
                on='features'
            )
            .merge(
                self.rf_features,
                how='left',
                on='features'
            )
            .merge(
                self.bfs_features,
                how='left',
                on='features'
            )
            .merge(
                self.ffs_features,
                how='left',
                on='features'
            )
        )
        self.summary['chosen'] = (
            self
            .summary
            .set_index('features')
            .notna()
            .sum(axis=1)
            .values
        )
        self.summary = self.summary[self.summary['features']!='dummy']
    
    def run_evaluation(self, X_train, y_train, wrapper_threshold=2, sample_size=0.2):
        """ Run every method and aggregate their reports"""
        self.X_train = X_train
        self.y_train = y_train
        self.wrapper_threshold = wrapper_threshold

        self.sample_dataset(sample_size=sample_size)

        self.X_train['dummy'] = np.random.random(self.X_train.shape[0])

        self.report = pd.DataFrame({'features': self.X_train.columns})
        
        self.information_filter()
        print('Info Filter Done!')
        self.lasso_embedded()
        print('Lasso Embedded Done!')
        self.rf_embedded()
        print('RF Importance Done!')
        self.compute_bfs_wrapper()
        print('BFS Done!')
        self.compute_ffs_wrapper()
        print('FFS Done!')
        self.parse_summary()
        
        return self.summary
    
    def selected_features(self, chosen=2):
        """ Return selected features depending o the number of methods that were chosen
        """
        
        self.select_features = (
            self.summary[self.summary['chosen']>=chosen]['features']
            .values
        )

        return self.select_features