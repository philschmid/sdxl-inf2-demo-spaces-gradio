import sagemaker
import boto3
from sagemaker.s3 import S3Uploader
import os
from huggingface_hub import snapshot_download
from distutils.dir_util import copy_tree
from sagemaker.huggingface.model import HuggingFaceModel
import tarfile
import shutil


os.environ["AWS_DEFAULT_REGION"] = "us-east-2"

save_directory = "sdxl_neuron"
compiled_model_id = "Jingya/lcm-sdxl-neuronx"

sess = sagemaker.Session()
# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it not exists
sagemaker_session_bucket = None
if sagemaker_session_bucket is None and sess is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sess.default_bucket()

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client("iam")
    role = iam.get_role(RoleName="sagemaker_execution_role")["Role"]["Arn"]

sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

print(f"sagemaker role arn: {role}")
print(f"sagemaker bucket: {sess.default_bucket()}")
print(f"sagemaker session region: {sess.boto_region_name}")
assert sess.boto_region_name in [
    "us-east-2",
    "us-east-1",
], "region must be us-east-2 or us-west-2, due to instance availability"


# Downloads our compiled model from the HuggingFace Hub
# using the revision as neuron version reference
# and makes sure we exlcude the symlink files and "hidden" files, like .DS_Store, .gitignore, etc.
snapshot_download(
    compiled_model_id,
    # revision="2.15.0",
    revision="main",
    local_dir=save_directory,
    local_dir_use_symlinks=False,
    allow_patterns=["[!.]*.*"],
)
copy_tree("code/", f"{save_directory}/code/")


# Create model.tar.gz
def compress(tar_dir=None, output_file="model.tar.gz"):
    parent_dir = os.getcwd()
    os.chdir(tar_dir)
    with tarfile.open(os.path.join(parent_dir, output_file), "w:gz") as tar:
        for root, dirs, files in os.walk("."):
            for file in files:
                file_path = str(os.path.join(root, file)).replace("./", "")
                print(file_path)
                tar.add(file_path, arcname=file_path)
    os.chdir(parent_dir)


compress(save_directory)

# create s3 uri
s3_model_path = f"s3://{sess.default_bucket()}/neuronx/lcm"

# upload model.tar.gz
s3_model_uri = S3Uploader.upload(
    local_path="model.tar.gz", desired_s3_uri=s3_model_path
)
s3_model_uri = "s3://sagemaker-us-east-2-558105141721/neuronx/lcm/model.tar.gz"
print(f"model artifcats uploaded to {s3_model_uri}")

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    model_data=s3_model_uri,  # path to your model.tar.gz on s3
    role=role,  # iam role with permissions to create an Endpoint
    transformers_version="4.34.1",  # transformers version used
    pytorch_version="1.13.1",  # pytorch version used
    py_version="py310",  # python version used
    model_server_workers=1,  # number of workers for the model server
)

# deploy the endpoint endpoint
predictor = huggingface_model.deploy(
    initial_instance_count=1,  # number of instances
    instance_type="ml.inf2.xlarge",  # AWS Inferentia Instance
    volume_size=100,
)
# ignore the "Your model is not compiled. Please compile your model before using Inferentia." warning, we already compiled our model.
print("Endpoint deployed")
print("Endpoint name: ", predictor.endpoint_name)
# print("delete temp code repository")
# shutil.rmtree(save_directory)
