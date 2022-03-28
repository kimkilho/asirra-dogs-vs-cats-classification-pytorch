# Reference 1: https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-pipeline-python-sdk

import os.path as osp

from azureml.core import (
    Workspace, Experiment, Dataset, Datastore,
    ComputeTarget, Environment, ScriptRunConfig
)
from azureml.data import OutputFileDatasetConfig
from azureml.data.datapath import DataPath
from azureml.core.compute import AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import (
    Pipeline, PipelineParameter
)


def get_or_create_compute_target(ws, use_gpu=False):
    # Create a ComputeTarget that represents the machine resource on which the pipeline will run
    cluster_name = 'gpu-cluster' if use_gpu else 'cpu-cluster'

    found = False
    # Check if this compute target already exists in the workspace
    cts = ws.compute_targets
    if cluster_name in cts and cts[cluster_name].type == 'AmlCompute':
        found = True
        print('Found existing compute target.')
        compute_target = cts[cluster_name]
    if not found:
        print('Creating a new compute target...')
        compute_config = AmlCompute.provisioning_configuration(
            vm_size='STANDARD_NC6' if use_gpu else 'STANDARD_D2_V2',
            # vm_priority='lowpriority', #optional
            max_nodes=4,
        )

        # Create the cluster
        compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    return compute_target


if __name__ == '__main__':
    # Configure workspace
    workspace = Workspace.from_config(path='./ws-config.json')

    compute_target = get_or_create_compute_target(workspace, use_gpu=True)
    # Can poll for a minimum number of nodes and for a specific timeout.
    # If no min_node_count is provided, it will use the scale settings for the cluster.
    compute_target.wait_for_completion(show_output=True)

    datastore = workspace.get_default_datastore()
    datastore_paths = [DataPath(datastore, 'asirra')]    # FIXME
    asirra_ds = Dataset.File.from_files(path=datastore_paths)

    # Create the configuration for training
    train_env = Environment.from_conda_specification(
        name='torch-env', file_path='./torch_conda_dependencies.yml'
    )

    # Specify the pipeline step
    script_folder = './train'
    asirra_ds_input = asirra_ds.as_named_input('asirra_ds_input')    # type: DatasetConsumptionConfig

    # Training configuration
    train_cfg = ScriptRunConfig(
        source_directory=script_folder,
        script='train.py',
        compute_target=compute_target,
        environment=train_env,
    )

    # Hyper-parameters for a training run
    num_epochs_param = PipelineParameter(name='num_epochs', default_value=10)

    train_step = PythonScriptStep(
        name='train step',
        arguments=['--data_root_dir', asirra_ds_input.as_mount(),
                   '--batch_size', 64,
                   '--num_epochs', num_epochs_param,
                   '--init_learning_rate', 0.0003],
        source_directory=train_cfg.source_directory,
        script_name=train_cfg.script,
        runconfig=train_cfg.run_config,
        allow_reuse=True,
    )

    # Create and run the pipeline
    pipeline = Pipeline(workspace, steps=[train_step])

    # Publish the training pipeline
    published_pipeline = pipeline.publish(
        name='asirra-ResNet50 trainer',
        description='Training pipeline for ResNet-50 model on Asirra dataset',
        continue_on_step_failure=True,
    )

    print('Published pipeline id: {}'.format(published_pipeline.id))

    # # Create an experiment to run the pipeline
    # exp = Experiment(workspace=workspace, name='asirra-ResNet50')
    # run = exp.submit(pipeline)
    # run.wait_for_completion(show_output=True)
