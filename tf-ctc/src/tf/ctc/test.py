try:
  import pytest
  if __name__ == '__main__':
    import os
    base = os.path.dirname(__file__)
    pytest.main([base, '--verbose'])
except ImportError:
  print('Install testing libs to run tests: `pip install tf-ctc[test]`')