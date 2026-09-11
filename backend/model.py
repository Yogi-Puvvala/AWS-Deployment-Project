import os
import boto3
from transformers import pipeline


BUCKET_NAME = "imdb-distillbert-classifier-bucket"
S3_PREFIX = "imdb-distillbert/"
LOCAL_MODEL_PATH = "/app/imdb-distillbert"


def download_model_from_s3():
    s3 = boto3.client("s3")

    os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(
        Bucket=BUCKET_NAME,
        Prefix=S3_PREFIX
    ):
        for obj in page.get("Contents", []):
            s3_key = obj["Key"]

            # Skip the folder itself
            if s3_key.endswith("/"):
                continue

            relative_path = s3_key[len(S3_PREFIX):]

            local_file = os.path.join(
                LOCAL_MODEL_PATH,
                relative_path
            )

            os.makedirs(
                os.path.dirname(local_file),
                exist_ok=True
            )

            # Don't download if file already exists
            if not os.path.exists(local_file):
                print(f"Downloading {s3_key}...")

                s3.download_file(
                    BUCKET_NAME,
                    s3_key,
                    local_file
                )


download_model_from_s3()


classifier = pipeline(
    "text-classification",
    model=LOCAL_MODEL_PATH,
    tokenizer=LOCAL_MODEL_PATH
)


def load_model():
    return classifier