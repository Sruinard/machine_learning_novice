# import required libraries
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute, Environment, Model
from azure.ai.ml import command, Input
from azure.ai.ml.constants import AssetTypes

# Enter details of your AzureML workspace
subscription_id = '0abb6ec5-9030-4b3f-af04-09183c688576'
resource_group = 'csu-nl-jax-haiku-rl'
workspace = 'ml_at_scale'
compute_target = "jaxhaiku-gpu"
env_name = "docker-with-jaxhaiku"

# connect to the workspace
ml_client = MLClient(DefaultAzureCredential(),
                     subscription_id, resource_group, workspace)

env_docker_conda = Environment(
    image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu20.04",
    conda_file="./environment.yaml",
    name=env_name,
    description="Environment created from a Docker image plus Conda environment.",
)
ml_client.environments.create_or_update(env_docker_conda)


try:
    ml_client.compute.get(compute_target)
except Exception:
    print("Creating a new gpu compute target...")
    compute = AmlCompute(
        name=compute_target, size="Standard_NC6_Promo", min_instances=0, max_instances=4
    )
    ml_client.compute.begin_create_or_update(compute)


# define the command
command_job = command(
    code="./src",
    command="python training_at_cloud_scale.py",
    environment=env_docker_conda,
    compute=compute_target,
    display_name="rl-agent"
)

# submit the command
returned_job = ml_client.jobs.create_or_update(command_job)
# # get a URL for the status of the job
print(returned_job.services["Studio"].endpoint)
print(returned_job.id)
print(returned_job.experiment_name)
print(returned_job.display_name)
print(returned_job.id.split("/")[-1])


file_model = Model(
    path=f"azureml://jobs/{returned_job.id.split('/')[-1]}/outputs/artifacts/lunar/dqn/",
    type=AssetTypes.CUSTOM_MODEL,
    name="lunar_dqn",
    description="Model created from job run file.",
)
ml_client.models.create_or_update(file_model)
