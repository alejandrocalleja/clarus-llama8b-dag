import json
import requests


from typing import Any, Optional
from IDS_templates.log import logger
from io import StringIO


class RestIDSConsumerConnector:
    def __init__(self):
        pass

    def post(self, url: str, msg: Any) -> Optional[requests.Response]:
        logger.debug("POST Request json = {} target_url = {}".format(msg, url))
        try:
            headers = {
                        "Content-Type": "application/json",
                         "Authorization": "Basic aWRzVXNlcjpwYXNzd29yZA=="
            }
            ret = requests.post(url, data=msg, headers=headers, verify= False, timeout=10)
            logger.debug("POST returned {}".format(ret))
            return ret
        except requests.RequestException as e:
            logger.error(
                "Request error {} on POST REQUEST to {}".format(type(e).__name__, url)
            )
            return None
        
    def get(self, url: str,timeout:int) -> Optional[requests.Response]:
        #logger.debug("GET Request json = {} target_url = {}".format(msg, url))
        try:
            headers = {
                        "Authorization": "Basic YXBpVXNlcjpwYXNzd29yZA=="
            }
            ret = requests.get(url, headers=headers, verify= False, timeout=timeout)
            """  logger.debug("GET returned {}".format(ret)) """
            return ret
        except requests.RequestException as e:
            logger.error(
                "Request error {} on GET REQUEST to {}".format(type(e).__name__, url)
            )
            return None
    
    def is_artifact_internal_registered_by_resource_title(self, resource_title,trainning_provider_ip):
        
        #Get execution core description
        response = self.get('https://'+trainning_provider_ip+':8091',10)
        if response is None or response.status_code != 200 :
            return None
        resourceId = self.get_resourceid_from_post_response_by_title(response,resource_title)
        if resourceId is None:
            return False
        else:
            return True

    
    
    def get_external_artifact_by_resource_title(self, resource_title,provider_ip,provider_port,trainning_dataapp_ip, trainning_dataapp_port):

        #Get resource with expected title
        body = self.create_body_post_request_resources(provider_ip,provider_port)
        response = self.post('https://'+trainning_dataapp_ip+':'+trainning_dataapp_port+'/proxy',body)
        if response is None or response.status_code != 200 :
            return None
        resourceId = self.get_resourceid_from_post_response_by_title(response,resource_title)
        logger.info ('ResourceId: ' + resourceId)

        #Get resource description and extract contract description
        body = self.create_body_post_request_resource_description(provider_ip,provider_port,resourceId)
        response = self.post('https://'+trainning_dataapp_ip+':'+trainning_dataapp_port+'/proxy',body)
        if response is None or response.status_code != 200 :
            return None
        contract_artifact = self.get_contract_artifact_from_post_response(response)
        contract_id = self.get_contract_id_from_post_response(response)
        contract_permission = self.get_contract_permission_from_post_response(response)
        contract_provider = self.get_contract_provider_from_post_response(response)
        logger.info ('contract_artifact: {} ' .format(contract_artifact) )
        logger.info ('contract_id: {} ' .format(contract_id) )
        logger.info ('contract_permission: {} ' .format(contract_permission) )
        logger.info ('contract_provider: {} ' .format(contract_provider) )

        #Get resource contract agreement
        body = self.create_body_post_request_resource_contract(provider_ip,provider_port,contract_artifact,contract_id,contract_permission,contract_provider)
        response = self.post('https://'+trainning_dataapp_ip+':'+trainning_dataapp_port+'/proxy',body)
        if response is None or response.status_code != 200 :
            return None
        contract_agreement = self.get_contract_agreement_from_post_response(response)
        transfer_contract = self.get_transfer_contract_from_post_response(response)
        logger.info ('contract_agreement: {} ' .format(contract_agreement) )
        logger.info ('transfer_contract: {} ' .format(transfer_contract) )

        #Sign contract agreement
        body = self.create_body_post_request_resource_contract_agreement(provider_ip,provider_port,contract_artifact,contract_agreement)
        response =self.post('https://'+trainning_dataapp_ip+':'+trainning_dataapp_port+'/proxy',body)
        #Check status successfull   
        if response is None or response.status_code != 200 :
            return None 
        #Get reource artifact        
        body = self.create_body_post_request_artifact(provider_ip,provider_port,contract_artifact,transfer_contract)
        response = self.post('https://'+trainning_dataapp_ip+':'+trainning_dataapp_port+'/proxy',body)
        if response is None or response.status_code != 200 :
            return None
        artifact_data = self.get_artifact_from_post_response(response)
        logger.info ('artifact_data downloaded ')
        return artifact_data
    
    
    def create_body_post_request_resources(self,provider_ip,provider_port):


        body = "{ "
        body = body + ' "multipart": "form" , "Forward-To": "https://'
        body = body + provider_ip +":" + provider_port + '/data" ,'
        body = body + ' "messageType": "DescriptionRequestMessage" '
        body = body + "} "
        return body
    
    def get_resourceid_from_post_response_by_title(self,postResponse:requests.Response, modelId):

        try:
            jsonData = postResponse.json()
                    
            resources = jsonData["ids:resourceCatalog"][0]["ids:offeredResource"]
            found = False
            # resource = '{ "ids:version": "1.0.0","@id": "https://w3id.org/idsa/autogen/textResource/1f047b05-e32c-4d55-8415-88892860a928","ids:title": [{"@value": "Wine Exp","@type": "http://www.w3.org/2001/XMLSchema#string"}]}'
            for resource in resources:
                #title = self.buscar_elemento_en_json(resource,"ids:title")
                title = resource["ids:title"]
                if title is not None:
                    if modelId in title[0]["@value"]:
                        found = True
                        resourceId = resource["@id"]
                        break
          
                    
            if found :
                return resourceId
            else:
                return None
            
        except Exception as e:
            logger.error(
                "Parsing error: {} ".format(e)
            )
            return None                   

    def create_body_post_request_resource_description(self,provider_ip,provider_port,resourceid):

        body = "{ "
        body = body + ' "multipart": "form" , "Forward-To": "https://'
        body = body + provider_ip +":" + provider_port + '/data" ,'
        body = body + ' "messageType": "DescriptionRequestMessage",'
        body = body + ' "requestedElement": ' + '"' + resourceid + '"'
        body = body + "} "
        return body

    def get_contract_artifact_from_post_response(self,postResponse:requests.Response):

        try:
            jsonData = postResponse.json()
                    
            contract_artifact = jsonData["ids:representation"][0]["ids:instance"][0]["@id"]
            return contract_artifact
            
        except Exception as e:
            logger.error(
                "Parsing error: {} ".format(e)
            )
            return None     


    def get_contract_id_from_post_response(self,postResponse:requests.Response):

        try:
           
            jsonData = postResponse.json()
                    
            contract_id = jsonData["ids:contractOffer"][0]["@id"]
            return contract_id 
            
        except Exception as e:
            logger.error(
                "Parsing error: {} ".format(e)
            )
            return None  


    def get_contract_permission_from_post_response(self,postResponse:requests.Response):

        try:

            jsonData = postResponse.json()
            contract_permission = jsonData["ids:contractOffer"][0]["ids:permission"][0]  
            return json.dumps(contract_permission)
        
        except Exception as e:
            logger.error(
                "Parsing error: {} ".format(e)
            )
            return None     

    def get_contract_provider_from_post_response(self,postResponse:requests.Response):

        try:
            
            jsonData = postResponse.json()
            contract_provider = jsonData["ids:contractOffer"][0]["ids:provider"]["@id"]   
            return contract_provider
        
        except Exception as e:
            logger.error(
                "Parsing error: {} ".format(e)
            )
            return None     


    def create_body_post_request_resource_contract(self,provider_ip,provider_port, contract_artifact,contract_id,contract_permission,contract_provider): 

        body = "{ "
        body = body + ' "multipart": "form" , "Forward-To": "https://'
        body = body + provider_ip +":" + provider_port + '/data" ,'
        body = body + ' "messageType": "ContractRequestMessage",'
        body = body + ' "requestedElement": ' + '"' + contract_artifact + '" ,'
        body = body + '  "payload": { ' 
        body = body + ' "@context": {"ids": "https://w3id.org/idsa/core/","idsc": "https://w3id.org/idsa/code/"},"@type": "ids:ContractRequest",   '
        body = body + ' "@id": ' + '"' + contract_id + '" ,'
        body = body + ' "ids:permission":[ '  + contract_permission + '  ],'
        body = body + ' "ids:provider":{ "@id": ' + '"' + contract_provider + '" },'
        body = body + ' "ids:obligation":[], '
        body = body + ' "ids:prohibition":[], ' 
        body = body + ' "ids:consumer":{ "@id": "http://w3id.org/engrd/connector/consumer" } '
        body = body + "} "
        body = body + "} "
        return body

    def get_contract_agreement_from_post_response(self,postResponse:requests.Response):

        try:
            
            contract_agreement = postResponse.json()
            return json.dumps(contract_agreement)              
            
        
        except Exception as e:
            logger.error(
                "Parsing error: {} ".format(e)
            )
            return None     

    def get_transfer_contract_from_post_response(self,postResponse:requests.Response):

        try:
            
            json = postResponse.json()              
            transfer_contract = json["@id"]
            return transfer_contract
        
        except Exception as e:
            logger.error(
                "Parsing error: {} ".format(e)
            )
            return None     

    def create_body_post_request_resource_contract_agreement(self,provider_ip,provider_port, contract_artifact,contract_agreement):   

        body = "{ "
        body = body + ' "multipart": "form" , "Forward-To": "https://'
        body = body + provider_ip +":" + provider_port + '/data" ,'
        body = body + ' "messageType": "ContractAgreementMessage",'
        body = body + ' "requestedArtifact": ' + '"' + contract_artifact + '" ,'
        body = body + '  "payload": ' + contract_agreement 
        body = body + "} "
        return body
 

    def create_body_post_request_artifact(self,provider_ip,provider_port, contract_artifact,transfer_contract):   

        body = "{ "
        body = body + ' "multipart": "form" , "Forward-To": "https://'
        body = body + provider_ip +":" + provider_port + '/data" ,'
        body = body + ' "messageType": "ArtifactRequestMessage",'
        body = body + ' "requestedArtifact": ' + '"' + contract_artifact + '" ,'
        body = body + ' "transferContract": ' + '"' + transfer_contract + '" '
        body = body + "} "
        return body    
    
    def get_artifact_from_post_response(self,postResponse:requests.Response):

        try:
            
            # artifact_data = postResponse.text
            artifact_data = StringIO(postResponse.content.decode('utf-8'))            
            
            return artifact_data
        
        except Exception as e:
            logger.error(
                "Parsing error: {} ".format(e)
            )
            return None     
