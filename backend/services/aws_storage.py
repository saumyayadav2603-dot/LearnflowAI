
import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def upload_file_to_s3(
    local_file_path: str,
    object_key: str,
    content_type: str | None = None,
) -> str:
    """Upload a local file to the configured S3 bucket."""

    bucket_name = os.getenv("S3_BUCKET_NAME")
    region = os.getenv("AWS_REGION", "ap-southeast-2")

    if not bucket_name:
        raise RuntimeError("S3_BUCKET_NAME is not configured")

    s3 = boto3.client("s3", region_name=region)

    extra_args = {}
    if content_type:
        extra_args["ContentType"] = content_type

    try:
        s3.upload_file(
            local_file_path,
            bucket_name,
            object_key,
            ExtraArgs=extra_args,
        )
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"S3 upload failed: {exc}") from exc

    return f"s3://{bucket_name}/{object_key}"
