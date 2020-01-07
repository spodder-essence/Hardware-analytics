import json
import sys
import logging

import numpy as np
import pandas as pd
from google.cloud import bigquery as bq
from google.oauth2 import service_account
from google_auth_oauthlib import flow
from googleapiclient import discovery
from socket import timeout

from attribution import Attribution, Markov
from query import Campaign, Query, ADHQueryError
from decorators import validate_config

logging.basicConfig(level=logging.INFO)

API_SCOPES = ['https://www.googleapis.com/auth/cloud-platform',
              'https://www.googleapis.com/auth/bigquery',
              'https://www.googleapis.com/auth/adsdatahub']
ADH_API = 'adsdatahub'
ADH_VERSION = 'v1'

def authenticate(auth_type,creds_path):
    """Gets credentials for Google API
    
    Parameters
    ----------
    auth_type : str
        service_key or oauth
    creds_path : str
        path to credentials file
    """

    if auth_type == 'service_key':
        credentials = service_account.Credentials.from_service_account_file(creds_path).with_scopes(API_SCOPES)
    elif auth_type == 'oauth':
        appflow = flow.InstalledAppFlow.from_client_secrets_file(creds_path,scopes=API_SCOPES)
        credentials = appflow.run_local_server()

    return credentials

def adh_query(service, bq_client, query: Query, overwrite=False):
    try:
        logging.info('Querying ADH.')
        query.create_adh_query(service)
        query.start_adh_query(service)
        num_tries = 0
        df=None
        while (num_tries < 2) and (df is None):
            try:
                logging.info('Query started in ADH. Now, retrieving ADH result. Attempt #: %d'%num_tries)
                df = query.retrieve_adh_results(service,bq_client,overwrite=overwrite)
            except timeout:
                msg = f'ADH Query connection timed out for {format} format.'
                if num_tries == 0: msg += ' Will retry once and then give up.'
                logging.warning(msg)
            num_tries += 1
    finally:
        logging.info('Deleting ADH Query.')
        query.delete(service)
    return df

@validate_config
def main(config):
    if config['auth']['service_key_path']:
        auth_type = 'service_key'
        creds_path = config['auth']['service_key_path']
    elif config['auth']['client_secret_path']:
        auth_type = 'oauth'
        creds_path = config['auth']['client_secret_path']

    with open(creds_path, encoding='utf-8') as creds:
        info = json.loads(creds.read())
        try:
            project_id = info['project_id']
        except KeyError:
            project_id = info['installed']['project_id']

    logging.info('Authenticating user.')
    credentials = authenticate(auth_type,creds_path)
    logging.info('User authenticated.')

    c = config['campaign']

    logging.info('Creating campaign object.')
    campaign = Campaign(c['id'],c['ids'],c['country_code'],c['start_date'],c['end_date'],c['activity_ids'],c['events'],advertiserId=c['advertiser_id'])
    logging.info('Creating new BigQuery client.')
    bq_client = bq.Client(project=project_id,credentials=credentials)

    source = config['source']
    eval_dimension = source['eval_dimension']
    time_zone = source['time_zone']
    query = Query(campaign, eval_dimension=eval_dimension, time_zone=time_zone)
    if source['query_format'] == 'ALL':
        formats = ['node','path']
    else:
        formats = [source['query_format']]
    if source['use'] == 'ALL':
        sources = ['ADH','BQ']
    else:
        sources = [source['use']]
    if 'ADH' in sources:
        developerKey = config['auth']['developer_key']
        adh_config = source['ADH']
        query.customer_id = adh_config['customer_id']
        query.data_customer_id = adh_config['data_customer_id']
        logging.info('Building ADH service.')
        #service = discovery.build(ADH_API, ADH_VERSION, developerKey=developerKey, credentials=credentials)
        ## New Code
        _FCQ_SERVICE = 'adsdatahub.googleapis.com'
        _FCQ_VERSION = 'v1'
        _DISCOVERY_URL_TEMPLATE = 'https://' +'%s/$discovery/rest?version=%s&key=%s'
        discovery_url = _DISCOVERY_URL_TEMPLATE % (_FCQ_SERVICE, _FCQ_VERSION, developerKey)
 
        service=discovery.build(ADH_API, ADH_VERSION, credentials=credentials,
                                discoveryServiceUrl=discovery_url, 
                                cache_discovery=False)        
        
        ## End of New Code
        
        customer_list=service.customers().list().execute()
        print(customer_list)
        
    if 'BQ' in sources:
        bq_config = source['BQ']
        query.dcm_account_id = bq_config['dcm_account_id']
        query.floodlight_config_ids = bq_config['floodlight_config_ids']
        logging.info(f'Pulling campaign impressions, uniques, and cost by {query.eval_dimension}')
        cm_info = query.bq_imps_cost(bq_client)
        logging.info(f'Saving campaign impressions, uniques and cost by {query.eval_dimension} to csv.')
        cm_info.to_csv(f"campaignInfo_{campaign.id}_{query.eval_dimension}.csv", index=False)

    results = {}
    for format in formats:
        query.format = format
        for item in sources:
            try:
                if item == 'ADH': df = adh_query(service, bq_client, query, adh_config['overwrite'])
                elif item == 'BQ':
                    logging.info('Querying BigQuery.')
                    df = query.bq_query(bq_client)
                else: raise NotImplementedError("The source provided is not supported. Choose between ADH and BQ.")
                results[query.title+'_'+item] = df
            except ADHQueryError as e:
                msg = f'ADH Query in {format} format failed. Error Message: ' + e.__repr__()
                logging.warning(msg)
                continue

    return results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = 'config.json'
    with open(config_path, encoding='utf-8') as config_file:
        config = json.loads(config_file.read())

    results = main(config)
    for name, df in results.items():
        logging.info(f'Saving results for {name} to csv.')
        df.to_csv(f'{name}.csv', index=False)
