import oss2


class OSSHub(object):
    """support model and dataset stored in oss"""

    def __init__(self, endpoint, access_key_id, access_key_secret, bucket_name):
        self.bucket_name = bucket_name
        self.bucket = oss2.Bucket(oss2.Auth(access_key_id, access_key_secret), endpoint, bucket_name)

    def have_model(self, model_id):
        result = self.bucket.list_objects(prefix=self.resource_to_oss_path("model", model_id))
        return len(result.object_list) > 1

    def have_dataset(self, dataset_id):
        result = self.bucket.list_objects(prefix=self.resource_to_oss_path("dataset", dataset_id))
        return len(result.object_list) > 1

    def model_size(self, model_id):
        """return the size of model in GB"""

        total_size = 0
        for obj in oss2.ObjectIterator(self.bucket, prefix=self.resource_to_oss_path("model", model_id)):
            total_size += obj.size

        # Convert bytes to GB
        size_in_gb = total_size / (1024 ** 3)
        return size_in_gb

    def dataset_size(self, dataset_id):
        """return the size of dataset in GB"""

        total_size = 0
        for obj in oss2.ObjectIterator(self.bucket, prefix=self.resource_to_oss_path("dataset", dataset_id)):
            total_size += obj.size

        # Convert bytes to GB
        size_in_gb = total_size / (1024 ** 3)
        return size_in_gb

    def model_url(self, model_id):
        """return the oss url by model id"""

        return "oss://%s/%s" % (self.bucket_name, self.resource_to_oss_path("model", model_id))

    def dataset_url(self, dataset_id):
        """return the oss by dataset id"""

        return "oss://%s/%s" % (self.bucket_name, self.resource_to_oss_path("dataset", dataset_id))

    def model_crd_name(self, model_id):
        return

    @staticmethod
    def resource_to_oss_path(resource_type, resource):
        if resource_type == "model":
            if resource.startswith("user-"):
                return "models/{}".format(resource)
            else:
                return "models/public/{}".format(resource)
        elif resource_type == "dataset":
            if resource.startswith("user-"):
                return "datasets/{}".format(resource)
            else:
                return "datasets/public/{}".format(resource)
        return ""
