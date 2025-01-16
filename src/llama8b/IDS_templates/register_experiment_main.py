import requests
import json
from uuid import uuid4
from datetime import datetime
from IDS_templates.log import logger
import os
# import sys

def getCatalog(ip, port):
    url = f'https://{ip}:{port}/'
    auth = ("apiUser", "password")

    try:
        response = requests.get(url, auth=auth, verify=False)
        response.raise_for_status()

        return response.json()
    except requests.exceptions.RequestException as err:
        return {"response": f"{err} Could not Fetch Data"}

def handle_post(filename, description, ids_ip):
    port = 8091
    data = getCatalog(ids_ip, port)
    catalog = data['ids:resourceCatalog'][0]['@id']
    url = f'https://{ids_ip}:{port}/api/offeredResource/'
    auth = ("apiUser", "password")
    headers = {
        "catalog": catalog,
    }

    json_path = os.path.join(os.path.dirname(__file__), '..', 'IDS_templates', 'json', 'offeredResource_template.json')

    with open(json_path, 'r') as template:
        offeredResource_template = json.load(template)

    new_path = 'mlflow:' + filename
    payload = json.dumps(offeredResource_template)
    payload = payload.replace('{{$guid}}', str(uuid4()))
    payload = payload.replace('{{$isoTimestamp}}', datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
    payload = json.loads(payload)
    payload["ids:representation"][0]["ids:instance"][0]["@id"] = new_path
    payload["ids:title"][0]["@value"] = filename
    payload["ids:description"][0]["@value"] = description

    try:
        response = requests.post(url, json=payload, headers=headers, auth=auth, verify=False)
        logger.debug("POST returned {}".format(response))
        response.raise_for_status()

        return {"response": response.json()}
    except requests.exceptions.RequestException as err:
        return {"response": f"{err} Could not Fetch Data"}

# def main():
#     handle_post('Test_file', 'This is a test', "172.16.56.43")

# if __name__ == "__main__":
#     main()
