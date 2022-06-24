import lightgbm as lgbm
import pandas as pd
import pytest
import xgboost as xgb

from feature_draft.api import FeatureDraft
from feature_draft import estimator, cross_val, feature_scouter


@pytest.fixture
def classification_data():
    return pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [2, 3, 4, 5],
        "c": [1, 0, 1, 0],
    })


@pytest.fixture
def regression_data():
    return pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [1, 2, 3, 4],
        "c": [1, 2, 3, 4],
    })


class TestFeatureDraft:

    @pytest.mark.parametrize(
        'model, model_type',
        [
            (lgbm.LGBMClassifier(), estimator.LightGBMEstimator),
            (lgbm.LGBMRegressor(), estimator.LightGBMEstimator),
            (xgb.XGBClassifier(), estimator.XGBoostEstimator),
            (xgb.XGBRegressor(), estimator.XGBoostEstimator),
        ],
    )
    def test_instantiation_with_model(
        self,
        model,
        model_type,
        classification_data,
    ):

        feature_draft = FeatureDraft(
            model=model,
            data=classification_data,
            features=["a", "b"],
            response="c",
        )

        assert isinstance(
            feature_draft.estimator,
            model_type,
        )

        for attr in [
            "data",
            "response",
            "candidate_features",
            "selected_features",
            "best_metrics",
        ]:
            assert hasattr(feature_draft, attr)

    @pytest.mark.parametrize(
        'model',
        [lgbm.LGBMClassifier(), xgb.XGBClassifier()],
    )
    def test_get_baseline_metric_classification(
        self,
        model,
        classification_data,
    ):

        feature_draft = FeatureDraft(
            model=model,
            data=classification_data,
            features=["a", "b"],
            response="c",
            cross_val_splits=5,
        )

        output = feature_draft._get_baseline_metric()

        assert output == [0.5] * 5

    @pytest.mark.parametrize(
        'model',
        [lgbm.LGBMRegressor(), xgb.XGBRegressor()],
    )
    def test_get_baseline_metric_regressor(
        self,
        model,
        regression_data,
    ):

        feature_draft = FeatureDraft(
            model=model,
            data=regression_data,
            features=["a", "b"],
            response="c",
            cross_val_splits=5,
        )

        output = feature_draft._get_baseline_metric()

        assert output == [1.25] * 5

    def test_build_with_candidate_feature(self, regression_data, mocker):

        mocker.patch.object(
            cross_val.CrossValidator,
            "cross_validation_build",
            return_value=1,
        )

        feature_draft = FeatureDraft(
            model=lgbm.LGBMRegressor(),
            data=regression_data,
            features=["a", "b"],
            response="c",
            cross_val_splits=5,
        )

        output = feature_draft._build_with_candidate_feature(feature="b")

        assert output == 1

        X = cross_val.CrossValidator.cross_validation_build.call_args[1]["X"]

        assert isinstance(X, pd.DataFrame)
        assert X.columns == ["b"]

    def test_update_feature_lists(self, regression_data):

        feature_draft = FeatureDraft(
            model=lgbm.LGBMRegressor(),
            data=regression_data,
            features=["a", "b"],
            response="c",
            cross_val_splits=5,
        )

        assert feature_draft.selected_features == []
        assert feature_draft.candidate_features == ["a", "b"]

        feature_draft._update_feature_lists(
            feature="b",
            feature_result=[2, 2, 2],
        )

        assert feature_draft.selected_features == ["b"]
        assert feature_draft.candidate_features == ["a"]
        assert feature_draft.best_metrics == [2, 2, 2]

    def test_draft_round(self, regression_data, mocker):

        mocker.patch.object(FeatureDraft, "_build_with_candidate_feature")
        mocker.patch.object(
            feature_scouter.Scout,
            "evaluate_results",
            return_value=1,
        )

        feature_draft = FeatureDraft(
            model=lgbm.LGBMRegressor(),
            data=regression_data,
            features=["a", "b"],
            response="c",
            cross_val_splits=5,
        )

        output = feature_draft.draft_round()

        assert output == 1
        assert FeatureDraft._build_with_candidate_feature.call_count == 2

    def test_draft_features_no_improvements(self, regression_data, mocker):

        mocker.patch.object(
            FeatureDraft,
            "draft_round",
            return_value=None
        )

        feature_draft = FeatureDraft(
            model=lgbm.LGBMRegressor(),
            data=regression_data,
            features=["a", "b"],
            response="c",
            cross_val_splits=5,
        )

        feature_draft.draft_features()

        assert feature_draft.selected_features == []
        assert feature_draft.candidate_features == ["a", "b"]
