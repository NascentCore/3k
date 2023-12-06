from modelscope.hub.api import HubApi


class ModelScopeHub(object):
    """Simple encapsulation of modelscope"""

    def download(self, model_id):
        pass

    def size(self, model_id):
        """return the size of model in GB"""

        total_size = 0
        file_list = HubApi().get_model_files(model_id)
        for file in file_list:
            size = file.get("Size", 0)
            total_size += size

        return total_size / 1024 / 1024 / 1024

    def git_url(self, model_id):
        """return the git url by model id"""

        return "https://www.modelscope.cn/%s.git" % model_id