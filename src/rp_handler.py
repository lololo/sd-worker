import time

import runpod
import requests
from requests.adapters import HTTPAdapter, Retry

LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"

automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))


# ---------------------------------------------------------------------------- #
#                              Automatic Functions                             #
# ---------------------------------------------------------------------------- #
def wait_for_service(url):
    '''
    Check if the service is ready to receive requests.
    '''
    while True:
        try:
            requests.get(url, timeout=120)
            return
        except requests.exceptions.RequestException:
            print("Service not ready yet. Retrying...")
        except Exception as err:
            print("Error: ", err)

        time.sleep(0.2)


def run_inference(inference_request):
    '''
    Run inference on a request.
    '''
    response = automatic_session.post(url=f'{LOCAL_URL}/txt2img',
                                      json=inference_request, timeout=600)
    return response.json()


def run(inference_request, method_name, http_method):
    print('------------------- RUN --------------------')
    print(inference_request)
    print(method_name)
    print(http_method)
    if http_method == 'POST':
        # response = automatic_session.post(url=f'{LOCAL_URL}/ 
        response = automatic_session.post(url=f'{LOCAL_URL}/{method_name}', json=inference_request, timeout=600)
        return response.json() 
    else:
        response = automatic_session.get(url=f'{LOCAL_URL}/{method_name}', json=inference_request, timeout=600)
        return response.json()  

# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
def handler(event):
    '''
    This is the handler function that will be called by the serverless.
    '''
    print('----------------')
    print(event)

    json = run(event["input"], event["input"]["name"], event["input"]["http"])



    # return the output that you want to be returned like pre-signed URLs to output artifacts
    return json


if __name__ == "__main__":
    wait_for_service(url=f'{LOCAL_URL}/txt2img')

    print("WebUI API Service is ready. Starting RunPod...")

    runpod.serverless.start({"handler": handler})
