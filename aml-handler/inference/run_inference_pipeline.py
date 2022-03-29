import os
import os.path as osp
import requests
import base64
import io

from PIL import Image

from azureml.core import (
    Workspace, Run,
)
from azureml.pipeline.core import PublishedPipeline
from azureml.core.authentication import InteractiveLoginAuthentication


if __name__ == '__main__':
    # Configure workspace
    workspace = Workspace.from_config(path='../ws-config.json')

    # Load published pipeline
    published_pipeline = [pipeline for pipeline in PublishedPipeline.list(workspace)
                          if pipeline.name == 'asirra-ResNet50 inference'][0]

    auth = InteractiveLoginAuthentication()
    aad_token = auth.get_authentication_header()

    rest_endpoint = published_pipeline.endpoint

    # Load image
    image_dir = osp.join(os.getcwd(), '..', '..', 'images')
    filename = 'cat.98.jpg'    # FIXME

    image_file_path = osp.join(image_dir, filename)
    im = Image.open(image_file_path)

    buffered = io.BytesIO()
    im.save(buffered, format='JPEG')
    img_b64 = base64.b64encode(buffered.getvalue())
    data = img_b64.decode('utf-8')

    # Specify the parameters when running the pipeline
    response = requests.post(
        rest_endpoint,
        headers=aad_token,
        json={
            'ExperimentName': 'asirra-ResNet50 inference exp',
            'RunSource': 'SDK',
            'ParameterAssignments': {'input_data': data},
        }
    )

    try:
        response.raise_for_status()
    except Exception:
        raise Exception(
            'Received bad response from the endpoint: {}\n'
            'Response code: {}\n'
            'Headers: {}\n'
            'Content: {}'.format(
                rest_endpoint, response.status_code, response.headers, response.content
            )
        )

    run_id = response.json().get('Id')
    print('Submitted pipeline run: ', run_id)

    run = Run.get(workspace, run_id)
    run.wait_for_completion(show_output=True)



