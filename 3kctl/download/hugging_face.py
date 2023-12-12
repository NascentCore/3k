from huggingface_hub import repo_exists, list_files_info


class HuggingFaceHub(object):
    """Simple encapsulation of huggingface"""

    def have_model(self, model_id):
        return repo_exists(model_id)

    def size(self, model_id):
        """return the size of model in GB"""

        total_size = 0
        file_list = list_files_info(model_id)
        for file in file_list:
            total_size += file.size

        return total_size / 1024 / 1024 / 1024

    def git_url(self, model_id):
        """return the git url by model id"""

        return "https://huggingface.co/%s" % model_id
