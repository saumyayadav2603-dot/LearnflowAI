
import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError


def start_knowledge_base_sync() -> dict:
    """Start syncing the configured S3 data source with Bedrock Knowledge Base."""

    region = os.getenv("AWS_REGION", "ap-southeast-2")
    knowledge_base_id = os.getenv("BEDROCK_KB_ID")
    data_source_id = os.getenv("BEDROCK_DATA_SOURCE_ID")

    if not knowledge_base_id:
        raise RuntimeError("BEDROCK_KB_ID is not configured")

    if not data_source_id:
        raise RuntimeError("BEDROCK_DATA_SOURCE_ID is not configured")

    bedrock_agent = boto3.client("bedrock-agent", region_name=region)

    try:
        response = bedrock_agent.start_ingestion_job(
            knowledgeBaseId=knowledge_base_id,
            dataSourceId=data_source_id,
        )
        return response["ingestionJob"]
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Bedrock Knowledge Base sync failed: {exc}") from exc
