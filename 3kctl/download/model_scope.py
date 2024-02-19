from modelscope.hub.api import HubApi
from modelscope.hub.errors import HTTPError, NotExistError


class ModelScopeHub(object):
    """Simple encapsulation of modelscope"""

    def have_model(self, model_id):
        try:
            model = HubApi().get_model(model_id)
            return True
        except (HTTPError, NotExistError) as e:
            return False

    def have_dataset(self, dataset_id):
        try:
            split_dataset = dataset_id.split('/')
            if len(split_dataset) != 2:
                return False
            dataset = HubApi().get_dataset_id_and_type(split_dataset[1], split_dataset[0])
            return True
        except (HTTPError, NotExistError) as e:
            return False

    def model_size(self, model_id):
        """return the size of model in GB"""

        total_size = 0
        file_list = HubApi().get_model_files(model_id, recursive=True)
        for file in file_list:
            size = file.get("Size", 0)
            total_size += size

        return total_size / 1024 / 1024 / 1024

    def dataset_size(self, dataset_id):
        """return the size of dataset in GB"""

        split_dataset = dataset_id.split('/')
        data_id, dataset_type = HubApi().get_dataset_id_and_type(split_dataset[1], split_dataset[0])
        total_size = 0
        file_list = HubApi().get_dataset_meta_file_list(split_dataset[1], split_dataset[0], data_id, "master")
        for file in file_list:
            size = file.get("Size", 0)
            total_size += size

        return total_size / 1024 / 1024 / 1024

    def model_url(self, model_id):
        """return the git url by model id"""

        return "https://www.modelscope.cn/%s.git" % model_id

    def dataset_url(self, dataset_id):
        """return the git url by dataset id"""

        return "https://www.modelscope.cn/datasets/%s.git" % dataset_id
