import subprocess
import unittest
import deploy.software as software
from unittest.mock import patch, mock_open, Mock


class TestLoadInstalledSoftwares(unittest.TestCase):

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data='["software1", "software2"]')
    def test_load_installed_softwares_exists(self, mock_file, mock_exists):
        mock_exists.return_value = True
        result = software.load_installed_softwares()
        self.assertEqual(result, {"software1", "software2"})

    @patch('os.path.exists')
    def test_load_installed_softwares_not_exists(self, mock_exists):
        mock_exists.return_value = False
        result = software.load_installed_softwares()
        self.assertEqual(result, set())


class TestInstallWithHelm(unittest.TestCase):

    @patch('subprocess.run')
    def test_install_with_helm_success(self, mock_run):
        mock_run.return_value = None  # 模拟成功执行
        try:
            software.install_with_helm("testapp", "default", None)
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    @patch('subprocess.run')
    def test_install_with_helm_failure(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(1, ['helm'])  # 模拟失败
        with self.assertRaises(Exception):
            software.install_with_helm("testapp", "default", None)


class TestApplyWithKubectl(unittest.TestCase):

    @patch('subprocess.run')
    def test_apply_with_kubectl_success(self, mock_run):
        # 模拟 subprocess.run 方法成功执行
        mock_run.return_value = None  # 表示命令成功执行
        try:
            software.apply_with_kubectl("test.yaml")
            mock_run.assert_called_with(["kubectl", "apply", "-f", "test.yaml"], check=True)
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    @patch('subprocess.run')
    def test_apply_with_kubectl_failure(self, mock_run):
        # 模拟 subprocess.run 方法抛出 CalledProcessError 异常
        mock_run.side_effect = subprocess.CalledProcessError(1, ['kubectl'])
        with self.assertRaises(subprocess.CalledProcessError):
            software.apply_with_kubectl("test.yaml")


class TestGetPodStatus(unittest.TestCase):

    @patch('subprocess.run')
    def test_get_pod_status_success(self, mock_run):
        # 模拟 subprocess.run 方法成功执行并返回期望的结果
        mock_run.return_value = Mock(returncode=0, stdout="Pod is ready", stderr="")
        result = software.get_pod_status("testapp", "default")
        self.assertIsNotNone(result)
        self.assertEqual(result.stdout, "Pod is ready")

    @patch('subprocess.run')
    def test_get_pod_status_failure(self, mock_run):
        # 模拟 subprocess.run 方法抛出 CalledProcessError 异常
        mock_run.side_effect = subprocess.CalledProcessError(1, ['kubectl-assert'], "Error occurred")
        result = software.get_pod_status("testapp", "default")
        self.assertIsNone(result)


class TestFindSoftwareByName(unittest.TestCase):

    def test_find_software_by_name_found(self):
        # 直接定义测试用的软件数据
        softwares = [
            {"name": "software1", "type": "helm"},
            {"name": "software2", "type": "kubectl"}
        ]
        softwares = softwares  # 设置模块中的软件列表

        sw = software.find_software_by_name("software1", softwares)
        self.assertEqual(sw, {"name": "software1", "type": "helm"})

    def test_find_software_by_name_not_found(self):
        # 同样直接定义测试用的软件数据
        softwares = [
            {"name": "software1", "type": "helm"},
            {"name": "software2", "type": "kubectl"}
        ]
        softwares = softwares  # 设置模块中的软件列表

        with self.assertRaises(ValueError) as context:
            software.find_software_by_name("software3", softwares)
        self.assertTrue("软件 software3 未定义" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
