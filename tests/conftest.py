import pytest
import warnings

@pytest.fixture(autouse=True)
def ignore_statistical_warnings(request):
    """
    Fixture to ignore expected runtime warnings from statistical functions 
    that intentionally test small sample sizes.
    
    This will not filter warnings in test functions that are specifically
    testing for warning behavior (test functions with 'warning' in their name).
    """
    # Don't filter warnings in test functions specifically testing for warnings
    if 'warning' in request.node.name.lower():
        yield
        return
        
    # Only filter warnings for other tests
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", 
                              message="Sample size.*is less than.*", 
                              category=RuntimeWarning)
        warnings.filterwarnings("ignore", 
                              message=".*may not be reliable for small samples", 
                              category=RuntimeWarning)
        warnings.filterwarnings("ignore", 
                              message="Mean of empty slice", 
                              category=RuntimeWarning)
        warnings.filterwarnings("ignore", 
                              message="Degrees of freedom <= 0 for slice", 
                              category=RuntimeWarning)
        yield 