import pytest
from DeepCNNClassifer.utils import read_yaml
from pathlib import Path
from box import ConfigBox
from ensure.main import EnsureError


class Test_read_yaml:
    yaml_files = [
        'tests/data/demo.yaml',
        'tests/data/empty.yaml'
    ]

    # Testing Error for empty yaml
    def test_read_empty_yaml(self):
        with pytest.raises(ValueError):
            read_yaml(Path(self.yaml_files[1]))

    # Testing return type of read_yaml()
    def test_read_yaml_return_type(self):
        response = read_yaml(Path(self.yaml_files[0]))
        assert isinstance(response, ConfigBox)

    # Testing the error for inappropriate return type
    @pytest.mark.parametrize("path_to_yaml", yaml_files)
    def test_read_yaml_bad_return_type(self, path_to_yaml):
        with pytest.raises(EnsureError):
            read_yaml(path_to_yaml)
