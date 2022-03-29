from azureml.core import (
    Workspace, Environment, ComputeTarget, ScriptRunConfig,
)
from azureml.data import OutputFileDatasetConfig
from azureml.core.compute import AmlCompute
from azureml.exceptions import ComputeTargetException
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import (
    Pipeline, PipelineParameter
)


def get_or_create_compute_target(ws, use_gpu=False):
    # Create a ComputeTarget that represents the machine resource on which the model will be running inference
    cluster_name = 'inference-gpu-cluster' if use_gpu else 'inference-cpu-cluster'

    # Check if this compute target already exists in the workspace
    try:
        compute_target = ComputeTarget(workspace=ws, name=cluster_name)
        print('Found existing compute target.')
    except ComputeTargetException:
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
    workspace = Workspace.from_config(path='../ws-config.json')

    compute_target = get_or_create_compute_target(workspace, use_gpu=True)
    compute_target.wait_for_completion(show_output=True)

    datastore = workspace.get_default_datastore()

    # 1. Specify the `process` step
    process_env = Environment.from_conda_specification(
        name='process-env', file_path='process_env_dependencies.yml'
    )
    process_cfg = ScriptRunConfig(
        source_directory='./scripts',
        script='process_input.py',
        compute_target=compute_target,
        environment=process_env,
    )
    input_data_param = PipelineParameter(name='input_data', default_value='')

    prepared_input = OutputFileDatasetConfig(
        destination=(datastore, 'processed_inputs/{run-id}')
    ).register_on_complete(name='prepared_input')

    process_step = PythonScriptStep(
        name='process step',
        arguments=['--input_data', input_data_param,
                   '--prepared_input_dir', prepared_input],
        source_directory=process_cfg.source_directory,
        script_name=process_cfg.script,
        runconfig=process_cfg.run_config,
        allow_reuse=True,
    )

    # 2. Specify the `predict` step
    predict_env = Environment.from_conda_specification(
        name='predict-env', file_path='predict_env_dependencies.yml'
    )
    predict_cfg = ScriptRunConfig(
        source_directory='./scripts',
        script='predict.py',
        compute_target=compute_target,
        environment=predict_env,
    )

    output = OutputFileDatasetConfig(
        destination=(datastore, 'outputs/{run-id}')
    ).register_on_complete(name='output')

    predict_step = PythonScriptStep(
        name='predict step',
        arguments=['--prepared_input_data', prepared_input.as_input(name='prepared_input_data'),
                   '--output_dir', output],
        source_directory=predict_cfg.source_directory,
        script_name=predict_cfg.script,
        runconfig=predict_cfg.run_config,
        allow_reuse=True,
    )

    # Create and run the pipeline
    pipeline = Pipeline(workspace, steps=[process_step, predict_step])

    # Publish the pipeline
    published_pipeline = pipeline.publish(
        name='asirra-ResNet50 inference',
        description='Inference pipeline for ResNet-50 model trained on Asirra dataset',
        continue_on_step_failure=True,
    )

    print('Published pipeline id: {}'.format(published_pipeline.id))
