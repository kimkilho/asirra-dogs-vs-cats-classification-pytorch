# Reference: https://github.com/Azure/azureml-examples/blob/main/python-sdk/tutorials/using-pipelines/publish-and-run-using-rest-endpoint.ipynb
import os.path as osp
import requests

from azureml.core import (
    Workspace, Run,
)
from azureml.pipeline.core import PublishedPipeline
from azureml.core.authentication import InteractiveLoginAuthentication


if __name__ == '__main__':
    # Configure workspace
    workspace = Workspace.from_config(path='../ws-config.json')

    # Load published pipeline
    # print(PublishedPipeline.list(workspace))
    pipeline_id = '348f2d00-3852-44a1-b08d-813e226e94ae'    # FIXME
    published_pipeline = PublishedPipeline.get(workspace, pipeline_id)

    auth = InteractiveLoginAuthentication()
    aad_token = auth.get_authentication_header()

    rest_endpoint = published_pipeline.endpoint

    # Specify the parameters when running the pipeline
    response = requests.post(
        rest_endpoint,
        headers=aad_token,
        json={
            'ExperimentName': 'asirra-ResNet50 training exp',
            'RunSource': 'SDK',
            'ParameterAssignments': {'num_epochs': 3},    # FIXME
        }
    )

    try:
        response.raise_for_status()
    except Exception:
        raise Exception(
            'Received bad response from the endpoint: {}\n'
            'Response Code: {}\n'
            'Headers: {}\n'
            'Content: {}'.format(
                rest_endpoint, response.status_code, response.headers, response.content
            )
        )

    run_id = response.json().get('Id')
    print('Submitted pipeline run: ', run_id)

    run = Run.get(workspace, run_id)
    run.wait_for_completion(show_output=True)

    model = run.find_step_run('train step')[0].register_model(
        model_name='asirra-ResNet50',
        model_path='outputs/asirra_ResNet50.pth',
    )
    # model = run.register_model(model_name='asirra-ResNet50',
    #                            model_path='outputs/asirra_ResNet50.pth')

    # print(model.name, model.id, model.version, sep='\t')
    print('Registered model id: {}'.format(model.id))
