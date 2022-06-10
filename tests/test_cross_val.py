import lightgbm as lgbm
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from feature_draft import cross_val, estimator


class TestCrossValidator:

    def test_instantiation(self):

        test_cv = cross_val.CrossValidator(n_splits=3)

        assert test_cv.n_splits == 3
        assert isinstance(test_cv.fold_object, StratifiedKFold)

    def test_cross_validation_build(self, mocker):

        data = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [2, 3, 4, 5, 6],
            "c": [0, 0, 1, 1, 1],
        })

        model = estimator.LightGBMEstimator(lgbm.LGBMClassifier())

        mocker.patch.object(
            estimator.LightGBMEstimator,
            "fit_and_evaluate",
            return_value=1,
        )

        test_cv = cross_val.CrossValidator(n_splits=3)

        output = test_cv.cross_validation_build(
            X=data[["a", "b"]],
            y=data["c"],
            model=model,
        )

        assert output == [1, 1, 1]
