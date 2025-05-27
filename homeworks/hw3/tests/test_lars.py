# import numpy as np
# from sklearn.metrics import r2_score
# from sklearn.linear_model import Lars as SklearnLars
# import import_ipynb  # noqa: F401
# from homework import LARS  # type: ignore


# def test_homework_lars():
#     X_train = np.array([[1], [2], [3], [4], [5]])
#     y_train = np.array([2, 3, 5, 7, 11])

#     model = LARS()
#     model.fit(X_train, y_train)
#     preds = model.predict(X_train)

#     assert preds.shape == (5,), "Predictions should match number of samples"
#     assert r2_score(y_train, preds) > 0.8, "R2 should be above 0.8"

#     sklearn_model = SklearnLars()
#     sklearn_model.fit(X_train, y_train)
#     sklearn_preds = sklearn_model.predict(X_train)

#     assert np.allclose(preds, sklearn_preds, rtol=1e-03), (
#         "Predictions should match sklearn's implementation"
#     )
import unittest
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import Lars as SklearnLars
import import_ipynb  # noqa: F401
from homework import LARS  # type: ignore


class TestHomeworkLars(unittest.TestCase):
    def test_lars_fit_predict(self):
        X_train = np.array([[1], [2], [3], [4], [5]])
        y_train = np.array([2, 3, 5, 7, 11])

        model = LARS()
        model.fit(X_train, y_train)
        preds = model.predict(X_train)

        self.assertEqual(preds.shape, (5,), "Predictions should match number of samples")
        self.assertGreater(r2_score(y_train, preds), 0.8, "R2 should be above 0.8")

        sklearn_model = SklearnLars()
        sklearn_model.fit(X_train, y_train)
        sklearn_preds = sklearn_model.predict(X_train)

        np.testing.assert_allclose(preds, sklearn_preds, rtol=1e-03,
                                   err_msg="Predictions should match sklearn's implementation")


if __name__ == '__main__':
    unittest.main()
