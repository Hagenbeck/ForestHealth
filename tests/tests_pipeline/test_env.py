import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def test_numpy():
    
    val = np.int64(18)
    py_val = int(val)
    
    assert py_val == 18
    assert isinstance(py_val, int)
    assert type(val).__name__ == 'int64'

def test_pandas():

    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    
    assert 'a' in df.columns
    assert df.shape == (2, 2)
    assert df['a'].sum() == 3

def test_sklearn():

    X = [[1], [2], [3], [4]]
    y = [2, 4, 6, 8]
    
    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict([[5]])

    assert abs(pred[0] - 10) < 1e-6