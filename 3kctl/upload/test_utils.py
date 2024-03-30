import unittest
import unittest.mock as mock
from unittest.mock import patch, Mock
from upload.utils import get_hashed_name, create_pvc, create_pod_with_pvc, copy_to_pvc, wait_for_pod_ready

class TestKubernetesOperations(unittest.TestCase):
    def test_get_hashed_name(self):
        name = "test-name"
        hashed = get_hashed_name(name)
        self.assertEqual(len(hashed), 16)
        self.assertIsInstance(hashed, str)
        
    @patch('upload.utils.config.load_kube_config')
    @patch('kubernetes.client.CoreV1Api')
    def test_create_pvc(self, mock_api, mock_load_kube_config):
        mock_load_kube_config.return_value = None  # 模拟 load_kube_config 方法

        namespace = 'default'
        pvc_name = 'test-pvc'
        size_in_bytes = 1024 ** 3  # 1 GiB

        create_pvc(namespace, pvc_name, size_in_bytes)

        mock_api().create_namespaced_persistent_volume_claim.assert_called_once()
        args, kwargs = mock_api().create_namespaced_persistent_volume_claim.call_args
        self.assertEqual(kwargs['namespace'], namespace)
        self.assertEqual(kwargs['body'].metadata.name, pvc_name)
        self.assertIn('1Gi', kwargs['body'].spec.resources.requests['storage'])


    @patch('upload.utils.config.load_kube_config')
    @patch('kubernetes.client.CoreV1Api')
    def test_create_pod_with_pvc(self, mock_api, mock_load_kube_config):
        mock_load_kube_config.return_value = None  # 模拟 load_kube_config 方法

        namespace = 'default'
        pod_name = 'test-pod'
        pvc_name = 'test-pvc'

        create_pod_with_pvc(namespace, pod_name, pvc_name)

        mock_api().create_namespaced_pod.assert_called_once()
        args, kwargs = mock_api().create_namespaced_pod.call_args
        self.assertEqual(kwargs['namespace'], namespace)
        self.assertEqual(kwargs['body'].metadata.name, pod_name)
        self.assertEqual(kwargs['body'].spec.volumes[0].persistent_volume_claim.claim_name, pvc_name)

    @patch('subprocess.run')
    @patch('os.listdir')
    def test_copy_to_pvc(self, mock_listdir, mock_subprocess_run):
        namespace = 'default'
        pod_name = 'test-pod'
        src_dir = '/path/to/src'
        mock_listdir.return_value = ['file1.txt', 'dir1']

        copy_to_pvc(namespace, pod_name, src_dir)

        mock_listdir.assert_called_with(src_dir)
        # Check if subprocess.run is called for each file/directory
        self.assertEqual(mock_subprocess_run.call_count, len(mock_listdir.return_value))

    @patch('upload.utils.config.load_kube_config')
    @patch('kubernetes.client.CoreV1Api')
    def test_wait_for_pod_ready(self, mock_api, mock_load_kube_config):
        mock_load_kube_config.return_value = None  # 模拟 load_kube_config 方法

        namespace = 'default'
        pod_name = 'test-pod'
        timeout = 30

        mock_api().read_namespaced_pod.return_value = Mock(
            status=Mock(phase="Running")
        )

        wait_for_pod_ready(namespace, pod_name, timeout)

        mock_api().read_namespaced_pod.assert_called_with(namespace=namespace, name=pod_name)


if __name__ == '__main__':
    unittest.main()
