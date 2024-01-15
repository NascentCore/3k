import unittest
from unittest.mock import mock_open, patch
from .utils import dict_to_obj, parse_yaml, parse_ini


class TestDictToObj(unittest.TestCase):

    def test_dict_to_obj(self):
        test_dict = {"a": 1, "b": 2}
        obj = dict_to_obj(test_dict)
        self.assertEqual(obj.a, 1)
        self.assertEqual(obj.b, 2)


class TestParseYaml(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data="a: 1\nb: 2")
    def test_parse_yaml(self, mock_file):
        result = parse_yaml()
        self.assertEqual(result.a, 1)
        self.assertEqual(result.b, 2)


class TestParseIni(unittest.TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data="[section]\nkey=value")
    def test_parse_ini(self, mock_file):
        config = parse_ini()
        self.assertEqual(config.section.key, 'value')


if __name__ == '__main__':
    unittest.main()