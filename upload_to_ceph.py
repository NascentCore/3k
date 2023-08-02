import os
import tarfile
import boto3

def create_tarfile(source_dir, output_filename):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def upload_to_ceph(ceph_access_key, ceph_secret_key, bucket_name, local_file, remote_file):
    session = boto3.session.Session()
    s3_client = session.client(
        "s3",
        endpoint_url="http://<CEPH_ENDPOINT>",
        aws_access_key_id=ceph_access_key,
        aws_secret_access_key=ceph_secret_key,
    )
    s3_client.upload_file(local_file, bucket_name, remote_file)

if __name__ == "__main__":
    # 替换为Ceph访问密钥和桶名称
    ceph_access_key = "CEPH_ACCESS_KEY"
    ceph_secret_key = "CEPH_SECRET_KEY"
    bucket_name = "BUCKET_NAME"
    # 替换为要上传的本地文档路径和文件名
    local_file_path = "path/to/local/document.txt"
    remote_file_name = "document.txt"
    # 打包本地文档
    tar_filename = f"{remote_file_name}.tar.gz"
    create_tarfile(local_file_path, tar_filename)
    # 上传到Ceph
    upload_to_ceph(ceph_access_key, ceph_secret_key, bucket_name, tar_filename, remote_file_name)
    # 清理临时打包文件
    os.remove(tar_filename)
    print("上传成功！")
