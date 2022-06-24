import lightgbm as lgbm
import numpy as np
import pandas as pd
import pytest
from sklearn import metrics
import xgboost as xgb

from feature_draft import estimator


class DummyClassificationEstimator:

    def __init__(self):
        pass

    def fit(self, X, y, **kwargs):
        self.fitted_ = True

    def predict_proba(self, X):
        return np.array([[0, 1]])


class DummyRegressionEstimator:

    def __init__(self):
        pass

    def fit(self, X, y, **kwargs):
        self.fitted_ = True

    def predict(self, X):
        return np.array([1])


# modeify BaseEstimator to be testable
estimator.BaseEstimator._estimator_config[DummyClassificationEstimator] = [
    np.add
]
estimator.BaseEstimator._estimator_config[DummyRegressionEstimator] = [np.add]


class TestBaseEstimator():
    """Tests for the feature_draft.estimator.BaseEstimator class"""

    def test_estimator_instantiation(self):

        _ = estimator.BaseEstimator(estimator=None)

    def test_estimator_setter_error(self):

        with pytest.raises(TypeError):
            _ = estimator.BaseEstimator(estimator="test")

    def test_metric_setter_error(self):

        with pytest.raises(ValueError):
            _ = estimator.BaseEstimator(estimator=None, metric="test")

    @pytest.mark.parametrize(
        "model, is_classification",
        [
            (DummyClassificationEstimator(), True),
            (DummyRegressionEstimator(), False),
        ],
    )
    def test_is_classification(self, model, is_classification):

        test_estimator = estimator.BaseEstimator(
            estimator=model,
        )

        assert test_estimator.is_classification() is is_classification

    def test_fit(self):

        test_estimator = estimator.BaseEstimator(
            estimator=DummyClassificationEstimator(),
        )

        test_estimator.fit(X=pd.DataFrame(), y=pd.Series())

        assert test_estimator.estimator.fitted_ is True

    @pytest.mark.parametrize(
        "model",
        [
            DummyClassificationEstimator(),
            DummyRegressionEstimator()
        ],
    )
    def test_predict(self, model):

        test_estimator = estimator.BaseEstimator(
            estimator=model,
        )

        output = test_estimator.predict(X=pd.DataFrame())

        assert output == np.array([1])

    def test_evaluate(self):

        test_estimator = estimator.BaseEstimator(
            estimator=DummyClassificationEstimator(),
        )

        output = test_estimator.evaluate(2, 5)

        assert output == 7

    def test_fit_and_evaluate(self):

        test_estimator = estimator.BaseEstimator(
            estimator=DummyClassificationEstimator(),
        )

        output = test_estimator.fit_and_evaluate(
            X=pd.DataFrame(),
            y=pd.Series(),
            X_val=None,
            y_val=pd.Series([1]),
        )

        assert len(output) == 1
        assert output[0] == 2


class TestLightGBMEstimator():
    """Tests for the feature_draft.estimator.LightGBMEstimator class"""

    @pytest.mark.parametrize(
        "model, metric",
        [
            (lgbm.LGBMClassifier(), metrics.roc_auc_score),
            (lgbm.LGBMRegressor(), metrics.mean_squared_error),
        ],
    )
    def test_estimator_instantiation(self, model, metric):

        test_estimator = estimator.LightGBMEstimator(estimator=model)

        assert test_estimator.metric == metric

    def test_callbacks_passed(self, mocker):

        test_estimator = estimator.LightGBMEstimator(
            estimator=lgbm.LGBMClassifier()
        )

        mocker.patch.object(estimator.BaseEstimator, "fit", return_value=1)

        test_estimator.fit(X=pd.DataFrame(), y=pd.Series())

        assert (
            "callbacks" in estimator.BaseEstimator.fit.call_args[1].keys()
        )


class TestXGBoostEstimator():
    """Tests for the feature_draft.estimator.XGBoostEstimator class"""

    @pytest.mark.parametrize(
        "model, metric",
        [
            (xgb.XGBClassifier(), metrics.roc_auc_score),
            (xgb.XGBRegressor(), metrics.mean_squared_error),
        ],
    )
    def test_estimator_instantiation(self, model, metric):

        test_estimator = estimator.XGBoostEstimator(estimator=model)

        assert test_estimator.metric == metric

    def test_verbose_passed(self, mocker):

        test_estimator = estimator.XGBoostEstimator(
            estimator=xgb.XGBClassifier()
        )

        mocker.patch.object(estimator.BaseEstimator, "fit", return_value=1)

        test_estimator.fit(X=pd.DataFrame(), y=pd.Series())

        assert (
            "verbose" in estimator.BaseEstimator.fit.call_args[1].keys()
        )


class TestBuildEstimator():
    """Tests for the feature_draft.estimator.build_estimator function"""

    @pytest.mark.parametrize(
        "model, model_class",
        [
            (None, estimator.BaseEstimator),
            (lgbm.LGBMClassifier(), estimator.LightGBMEstimator),
            (lgbm.LGBMRegressor(), estimator.LightGBMEstimator),
            (xgb.XGBClassifier(), estimator.XGBoostEstimator),
            (xgb.XGBRegressor(), estimator.XGBoostEstimator),
        ],
    )
    def test_construction(self, model, model_class):

        output = estimator.build_estimator(model=model)

        assert isinstance(output, model_class)

    def test_error(self):

        with pytest.raises(TypeError):
            _ = estimator.build_estimator(model=1)
