from minio import Minio
import json
from processor.config_plus.minio_config import minio_config

try:
    # 创建MinIO客户端
    minio_client = Minio(
        endpoint=minio_config.endpoint,
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
        secure=False    #False：http访问；True：https访问
    )

    # 判断是否有对应的bucket，没有则创建
    if not minio_client.bucket_exists(minio_config.bucket_name):
        minio_client.make_bucket(minio_config.bucket_name)

    # 设置访问权限Custom
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": ["*"]},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{minio_config.bucket_name}/*"]
            }
        ]
    }
    minio_client.set_bucket_policy(minio_config.bucket_name, json.dumps(policy))
except Exception as e:
    print(f"Minio客户端初始化失败:{e}")

def get_minio_client():
    return minio_client


