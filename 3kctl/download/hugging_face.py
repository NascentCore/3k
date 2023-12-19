from huggingface_hub import repo_exists, list_files_info


class HuggingFaceHub(object):
    """Simple encapsulation of huggingface"""

    def have_model(self, model_id):
        return repo_exists(repo_id=model_id)

    def have_dataset(self, dataset_id):
        return repo_exists(repo_id=dataset_id, repo_type="dataset")

    def model_size(self, model_id):
        """return the size of model in GB"""

        total_size = 0
        file_list = list_files_info(model_id)
        for file in file_list:
            total_size += file.size

        return total_size / 1024 / 1024 / 1024

    def dataset_size(self, dataset_id):
        """return the size of dataset in GB"""

        total_size = 0
        file_list = list_files_info(dataset_id, repo_type="dataset")
        for file in file_list:
            total_size += file.size

        return total_size / 1024 / 1024 / 1024

    def git_model_url(self, model_id):
        """return the git url by model id"""

        return "https://huggingface.co/%s" % model_id

    def git_dataset_url(self, dataset_id):
        """return the git url by model id"""

        return "https://huggingface.co/datasets/%s" % dataset_id
