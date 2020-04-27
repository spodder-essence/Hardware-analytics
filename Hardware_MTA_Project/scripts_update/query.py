"""
Author: Drew Lustig, Yvette Wang, Maximilian Schmitt
Please feel free to contact me for questions about this module (maximilian.schmitt@)
"""

import os
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from google.cloud import bigquery as bq
from googleapiclient import errors
from google.api_core.exceptions import NotFound


class QueryExistsError(Exception):
    pass


class ADHQueryError(Exception):
    pass


class Campaign(object):
    """Class used to represent a DCM Campaign

    Parameters
    ----------
    id : List[int]
        DCM Campaign ID
    ids : List[int]
        List of DCM Campaign Search ID(s)
    country_code : List[String]
        List of DCM Country Code(s)
    startDate : str (yyyy-mm-dd)
        Date the campaign started
    endDate : str (yyyy-mm-dd)
        Date the campaign ended
    activityIds : List[int]
        List of activity ids to count as conversions
    events : List[Dict]
        List of event start and end times
        ex. [{"start_timestamp": "yyyy-mm-dd hh:mm:ss",
            "end_timestamp": "yyyy-mm-dd hh:mm:ss"}]
    advertiserId : List[int]
        DCM advertiser ID of campaign
    """

    def __init__(
        self,
        id,
        ids,
        country_code,
        startDate,
        endDate,
        activityIds,
        events,
        advertiserId=None
    ):
        self.id = id
        self.ids = ids
        self.country_code = country_code
        self.startDate = datetime.strptime(startDate, "%Y-%m-%d").date()
        self.endDate = datetime.strptime(endDate, "%Y-%m-%d").date()
        self.advertiserId = advertiserId
        self.activityIds = activityIds
        self.events = events

    def __repr__(self):
        return f'<Campaign {self.id}: startDate={self.startDate}, endDate={self.endDate}, advertiserId={self.advertiserId}>'


class ADH(object):
    """Class that's essentially a wrapper for the ADH API
    
    Parameters
    ----------
    customer_id : int, optional
        ADH customer id
    time_zone : str, optional
        Time zone of dates and timestamps used in the query
    title : str, optional
        Title of the query
    dest_table : str, optional
        Name of the BigQuery destination table for the query.
        Must have this format: `dataset.table_name`
    start_date : str, optional
        Start date of the query date range.
    end_date : str, optional
        End date of the query date range.
    date_customr_id : int, optional
        ADH customer id of the data location, if different from the query customer id
    query_name : str, optional
        ADH-generated name of the query
    operation_name : str, optional
        ADH-generated name of the query operation

    Attributes
    ----------
    customer_id : int
    time_zone : str
    title : str
    dest_table : str
    start_date : str
    end_date : str
    data_customer_id : int
    query_name : str
    operation_name : str    
    """

    def __init__(self, customer_id=None, time_zone=None, title=None,
        dest_table=None, start_date=None, end_date=None, data_customer_id=None,
        query_name=None, operation_name=None):
        
        self.customer_id = customer_id
        self.time_zone = time_zone
        if title is not None:
            self.title = title
        if dest_table is not None:
            self.dest_table = dest_table
        self.start_date = start_date
        self.end_date = end_date
        self._data_customer_id = data_customer_id
        self.query_name = query_name
        self.operation_name = operation_name

    @property
    def data_customer_id(self):
        if self._data_customer_id is None:
            return self.customer_id
        else:
            return self._data_customer_id

    @data_customer_id.setter
    def data_customer_id(self, value):
        self._data_customer_id = value

    def create(self, service, sql, parameter_types=None):
        """Creates an ADH query and inserts it into ADH.

        Parameters
        ----------
        service : googleapiclient.discovery.Resource
            ADH API service resource
        sql : str
            SQL query to be executed
        parameter_types : dict, optional
            Parameter types, if sql query has parameters
            For more information, visit the ADH API documentation:
                https://developers.google.com/ads-data-hub/reference/rest/v1/ParameterType

        Returns
        -------
        query_name : str
            ADH-generated name of query

        Raises
        ------
        QueryExistsError
            If query already exists in ADH
        """

        body = {
            'title': self.title,
            'queryText': sql
        }

        if parameter_types:
            body['parameterTypes'] = parameter_types

        try:
            response = service.customers().analysisQueries().create(
                parent=f'customers/{self.customer_id}',
                body=body).execute()
        except errors.HttpError as e:
            if 'already exists' in e.__repr__():
                msg = 'Query already exists in ADH. Delete this query before creating a new one.'
                raise QueryExistsError(msg)
            else:
                raise e
        self.query_name = response['name']
        return self.query_name

    def start(self, service, parameter_values=None):
        """Starts a query job in ADH

        Parameters
        ----------
        service : googleapiclient.discovery.Resource
            Google resource object built for accessing ADH API
        parameter_values : dict, optional
            Values of any parameters to be used in the query
            For more information, visit the ADH API documentation:
                https://developers.google.com/ads-data-hub/reference/rest/v1/ParameterValue

        Returns
        -------
        operation_name : str
            ADH-generated name of the query operation
        """

        body = {
            'spec': {
                'adsDataCustomerId': self.data_customer_id,
                'startDate': {
                    'year': str(self.start_date.year),
                    'month': str(self.start_date.month),
                    'day': str(self.start_date.day)
                },
                'endDate': {
                    'year': str(self.end_date.year),
                    'month': str(self.end_date.month),
                    'day': str(self.end_date.day)
                },
                'timeZone': self.time_zone
            },
            'destTable': self.dest_table,
            'customerId': self.customer_id
        }

        if parameter_values:
            body['spec']['parameterValues'] = parameter_values

        response = service.customers().analysisQueries().start(
            name=self.query_name,
            body=body).execute()
        self.operation_name = response['name']
        return self.operation_name

    def results_available(self, bq_client):
        """Returns True if results of ADH query exist in BigQuery

        Parameters
        ---------
        bq_client : google.cloud.bigquery.Client
        """

        dataset_ref = bq_client.dataset(
            'analytics_results',
            project='essence-ads-data-hub')
        table_ref = dataset_ref.table(self.title)
        try:
            bq_client.get_table(table_ref)
        except NotFound:
            return False

        return True

    def to_dataframe(self, service, bq_client):
        """Returns results of ADH query as a DataFrame"""

        job = bq_client.query(
            f"SELECT * FROM {self.dest_table}",
            project='essence-ads-data-hub')
        result = job.result()
        df = result.to_dataframe()
        return df

    def result(self, service):
        """Waits for a started query operation to finish and returns operation metadata

        Parameters
        ----------
        service : googleapiclient.discovery.Resource
            Google resource object built for accessing ADH API

        Returns
        -------
        operation_data : dict
            Metadata of ADH operation

        Raises
        ------
        ADHQueryError
            If the query fails once pushed to ADH,
            shows the error message provided by ADH
        """

        query_finished = False
        while not query_finished:
            time.sleep(5)
            operation_data = service.operations().get(
                name=self.operation_name).execute()
            query_finished = 'done' in operation_data.keys()

        if 'error' in operation_data.keys():
            raise ADHQueryError(operation_data['error']['message'])

        return operation_data

    def delete(self, service):
        """Deletes query from ADH and returns ADH name of deleted query.

        Parameters
        ----------
        service : googleapiclient.discovery.Resource
            Google resource object built for accessing ADH API

        Returns
        -------
        name : str
            Name of deleted query
        """

        delete = ''
        query_name = self.query_name
        if query_name is None:
            response = service.customers().analysisQueries().list(
                parent=f'customers/{self.customer_id}',
                filter=f'title = {self.title}'
            ).execute()
            if 'queries' in response.keys():
                query = response['queries'][0]
                query_name = query['name']
                owner = query['createEmail']
                delete = input(f"Delete existing query owned by {owner}? Y/N: ").lower()
                while delete not in ['y', 'n']:
                    print('Please respond with Y or N')
                    delete = input(f"Delete existing query owned by {owner}? Y/N: ").lower()

        if self.query_name is not None or delete == 'y':
            service.customers().analysisQueries().delete(
                name=query_name).execute()
            self.query_name = None
            self.operation_name = None
        else:
            query_name = None

        return query_name


class Query(ADH):
    """Class used to represent a query

    The query will return campaign data in a format
    that can be used for attribution. The 'node' query format
    can only be used for Markov and Last-Touch, but is computationally
    faster and will return more results from ADH. The 'path' format
    can be used for any of the supported attribution methodologies,
    but risks omitting more data from ADH.

    Parameters
    ----------
    campaign : Campaign
    eval_dimension : str, optional
        Dimension to evaluate conversions on (i.e. 'placement_id')
        Must match a column in the logs source you plan to query.
    format : str, optional
        Format results will be in. 'path' or 'node' (Default is node)
        'node' returns more results from ADH and is faster,
            but can only be used for Markov and Last-Touch.
        'path' is slower and returns less results from ADH,
            but can be used for any methodology.
    time_zone : str, optional
        Default is 'US/Eastern'
    customer_id : int, optional
        ADH Customer ID (default is 48)
    data_customer_id : int, optional
        ADH Customer ID of data location, if it is different from customer_id
    dcm_account_id : int, optional
        DCM Account ID of advertiser. Required for BQ
    floodlight_config_ids : List[int], optional
        DCM Floodlight Configuration IDs of activities
    adh_query_name : str, optional
        ADH-generated name of query, if it already exists.
    adh_operation_name : str, optional
        ADH-generated name of query job (operation), if it already exists.

    Attributes
    ----------
    campaign
    eval_dimension
    format
    time_zone
    customer_id
    data_customer_id
    dcm_account_id
    floodlight_config_id
    start_date : date
        Start date of campaign
    end_date : date
        End date of campaign
    title : str
        Title of the query
    dest_table : str
        Destination table for ADH queries
    adh_text : str
        SQL query for ADH
    bq_imps_table : str
        Name of DT impressions table in BigQuery
    bq_attr_text : str
        SQL query for BigQuery
    """

    def __init__(self, campaign: Campaign, eval_dimension, format='node', time_zone='US/Eastern',
        customer_id=48, data_customer_id=None, dcm_account_id=None, floodlight_config_ids=None,
        adh_query_name=None, adh_operation_name=None):
        self.campaign = campaign
        self.eval_dimension = eval_dimension
        self._format = format
        self.dcm_account_id = dcm_account_id
        self.floodlight_config_ids = floodlight_config_ids
        self.uniqueID="GoogleNA_Hardware_MTA_"+str(int(time.time()))+"_"
        ADH.__init__(
            self,
            customer_id=customer_id,
            time_zone=time_zone,
            data_customer_id=data_customer_id,
            start_date=campaign.startDate,
            end_date=campaign.endDate,
            query_name=adh_query_name, 
            operation_name=adh_operation_name)

    @property
    def format(self):
        return self._format

    @format.setter
    def format(self, value):
        if value not in ['path', 'node']:
            raise ValueError(f'format must be either path or node.')
        self._format = value

    @property
    def title(self):
        eval_dim = self.eval_dimension.replace("_", "")
        return f'MarkovChainAttr_{eval_dim}_{self.format}'

    @property
    def dest_table(self):
        return f'analytics_results.{self.title}'

    @property
    def timezone_offset(self):
        timezone = {
            'US/Eastern': -4,
            'US/Pacific': -7,
            'US/Central': -5
        }
        try:
            timezone_offset = timedelta(hours=timezone[self.time_zone])
        except KeyError:
            raise NotImplementedError(f'Time zone {self.time_zone} is not yet supported for BigQuery. Please contact maximilian.schmitt@essenceglobal.com to have it added.')
        return timezone_offset


    @property
    def adh_text(self):
        """Returns sql string to be inserted into ADH"""

        event_range = range(len(self.campaign.events))
        event_list = [f"timestamp_micros(event.event_time) between @event_start{i} and @event_end{i}" for i in event_range]
        event_stmts = "(" + " or ".join(event_list) + ")"

        window_stmts = f"""
	Create Table {self.uniqueID+"imps"} AS(
            with clicks as (SELECT
            {self.eval_dimension}, 
            user_id, 
            event.event_time as event_time, 
            ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event.event_time ASC) AS rownum 
            from `adh.cm_dt_clicks` left join 
            (SELECT paid_search_legacy_keyword_id,
        
            CASE WHEN UPPER(paid_search_campaign) LIKE '%AW SEM%' AND UPPER(paid_search_campaign) LIKE '%BKWS%' AND campaign_id = 10413728 THEN "Nest_Google_BKWS"
            WHEN UPPER(paid_search_campaign) LIKE '%BING SEM%' AND UPPER(paid_search_campaign) LIKE '%BKWS%' AND campaign_id = 10413728 THEN "Nest_Bing_BKWS"
            WHEN UPPER(paid_search_campaign) LIKE '%AW SEM%' AND UPPER(paid_search_campaign) LIKE '%SKWS%' AND campaign_id = 10413728 THEN "Nest_Google_SKWS"
            WHEN UPPER(paid_search_campaign) LIKE '%BING SEM%' AND UPPER(paid_search_campaign) LIKE '%SKWS%' AND campaign_id = 10413728 THEN "Nest_Bing_SKWS"
            WHEN UPPER(paid_search_campaign) LIKE '%AW SEM%' AND UPPER(paid_search_campaign) LIKE '%BKWS%' AND campaign_id = 8582844 THEN "Store_Google_BKWS"
            WHEN UPPER(paid_search_campaign) LIKE '%BING SEM%' AND UPPER(paid_search_campaign) LIKE '%BKWS%' AND campaign_id = 8582844 THEN "Store_Bing_BKWS"
            WHEN UPPER(paid_search_campaign) LIKE '%AW SEM%' AND UPPER(paid_search_campaign) LIKE '%SKWS%' AND campaign_id = 8582844 THEN "Store_Google_SKWS"
            WHEN UPPER(paid_search_campaign) LIKE '%BING SEM%' AND UPPER(paid_search_campaign) LIKE '%SKWS%' AND campaign_id = 8582844 THEN "Store_Bing_SKWS"
            ELSE "OTHER" END AS Site_Tactic,
        
            CASE WHEN UPPER(paid_search_campaign) LIKE '%AW SEM%' AND UPPER(paid_search_match_type) LIKE '%EXACT%' THEN "Google_EXACT"
            WHEN UPPER(paid_search_campaign) LIKE '%BING SEM%' AND UPPER(paid_search_match_type) LIKE '%EXACT%' THEN "Bing_EXACT"
            WHEN UPPER(paid_search_campaign) LIKE '%AW SEM%' AND UPPER(paid_search_match_type) LIKE '%BROAD%'THEN "Google_BROAD"
            WHEN UPPER(paid_search_campaign) LIKE '%BING SEM%' AND UPPER(paid_search_match_type) LIKE '%BROAD%' THEN "Bing_BROAD"
            ELSE "OTHER" END AS Targeting,
            
            CASE WHEN UPPER(paid_search_campaign) LIKE '%BKWS%' THEN "SEM_BKWS"
            WHEN UPPER(paid_search_campaign) LIKE '%SKWS%'THEN "SEM_SKWS"
            ELSE "OTHER" END AS Tactic,
      
            count(*) as clicks
        
            FROM `adh.cm_dt_paid_search` group by 1,2,3,4) a 
            on a.paid_search_legacy_keyword_id = event.segment_value_1                      
            WHERE
                event.advertiser_id in UNNEST(@advertiser_Id)
                and event.campaign_id in UNNEST(@search_campaign_id)
                and user_id != '0'
                AND event.country_code in UNNEST(@country_code)
                And a.{self.eval_dimension} IS NOT NULL
            group by 1, 2, 3),
            
            
            imp as (select 
                user_id, 
                temp.{self.eval_dimension}, 
                event.event_time as event_time,
                ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event.event_time ASC) AS rownum 
                FROM `adh.cm_dt_impressions` LEFT JOIN
                (SELECT placement.placement_id AS p_id, placement, site.site, 
        
                CASE WHEN placement.placement LIKE '%NonRMKT%' AND campaign_id = 23558368 THEN CONCAT("Nest_", site.site, "_NonRMKT")
		WHEN placement.placement LIKE '%RMKT%' AND campaign_id = 23558368 THEN CONCAT("Nest_", site.site, "_RMKT")
		WHEN placement.placement LIKE '%NonRMKT%' AND campaign_id = 23540338 THEN CONCAT("Store_", site.site, "_NonRMKT")
                WHEN placement.placement LIKE '%RMKT%' AND campaign_id = 23540338 THEN CONCAT("Store_", site.site, "_RMKT")
                WHEN UPPER(placement.placement) LIKE '%TEST%' THEN CONCAT(site.site, "_Test")
                ELSE "OTHER" END AS Site_Tactic,
        
                CASE WHEN UPPER(placement.placement) LIKE '%IMS%' THEN "IMS"
                WHEN UPPER(placement.placement) LIKE '%AFFINITY%' THEN "Affinity"
                WHEN UPPER(placement.placement) LIKE '%CTXT%' THEN "CTXT"
                WHEN UPPER(placement.placement) LIKE '%PLACEMENT_TARGETING%' THEN "Placement_Targeting"
                ELSE "OTHER" END AS Targeting,
        
                CASE WHEN placement.placement LIKE '%NonRMKT%' THEN "NonRMKT"
                WHEN placement.placement LIKE '%RMKT%' THEN "RMKT"
                WHEN UPPER(placement.placement) LIKE '%TEST%' THEN "Test"
                ELSE "OTHER" END AS Tactic
               
                FROM `adh.cm_dt_placement` AS placement
        
                LEFT JOIN `adh.cm_dt_site` site
                ON placement.site_id = site.site_id) AS temp
                ON event.placement_id = temp.p_id
                    
                WHERE 
                    event.advertiser_id in UNNEST(@advertiser_Id)
                    AND event.campaign_id in UNNEST(@campaign_id)
                    AND user_id != '0'
                    AND event.country_code in UNNEST(@country_code)
                    AND {self.eval_dimension} IS NOT NULL
                group by 1, 2, 3),
            imps as (select
                 user_id,
                 temp.{self.eval_dimension},
                 event_time,
                 ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_time ASC) AS rownum
                 from (select
                       imp.user_id as user_id,
                       imp.{self.eval_dimension} as {self.eval_dimension},
                       imp.event_time as event_time,
                       imp.rownum as rownum from imp
                       union all
                       select
                       clicks.user_id as user_id,
                       clicks.{self.eval_dimension} as {self.eval_dimension},
                       clicks.event_time as event_time,
                       clicks.rownum as rownum from clicks) as temp
            group by 1, 2, 3)
	    select * from imps);

            CREATE TABLE {self.uniqueID+"activities"} as (
            select
                user_id,
                min(event.event_time) as min_time,
		max(event.event_time) as max_time,
                event.event_sub_type as event_sub_type
            from `adh.cm_dt_activities`
            where
                {event_stmts}
                and user_id != '0'
                and event.activity_id in UNNEST(@activity_ids)
            group by 1,4);


            CREATE TABLE {self.uniqueID+"exposed_nulls"} AS (
            (select
                user_id, min(max_row) as max_row, max(converter) as converter
            from (
                select user_id, max(rownum) as max_row, 0 as converter
                from {"tmp."+self.uniqueID+"imps"} i group by 1, 3
                union all
                select i.user_id, max(rownum) as max_row, 1 as converter
                from {"tmp."+self.uniqueID+"imps"} i join {"tmp."+self.uniqueID+"activities"} a
                on i.user_id = a.user_id
                    and i.event_time < a.min_time
                where a.event_sub_type = 'POSTCLICK'
                group by 1, 3
                union all
                select i.user_id, max(rownum) as max_row, 1 as converter
                from {"tmp."+self.uniqueID+"imps"} i join {"tmp."+self.uniqueID+"activities"} a
                on i.user_id = a.user_id
                    and i.event_time < a.min_time
                where a.event_sub_type = 'POSTVIEW'
                group by 1, 3)
            group by 1));

CREATE TABLE {self.uniqueID+"exposed_nulls2"} as (
            (select
                user_id, max_row, min_row, max(converter) as converter
            from (
                select i.user_id, max(rownum) as max_row, min(rownum) as min_row, 1 as converter
                from {"tmp."+self.uniqueID+"imps"} i join {"tmp."+self.uniqueID+"activities"} a
                on i.user_id = a.user_id
                    and i.event_time > a.min_time
                    and i.event_time < a.max_time
                where a.event_sub_type = 'POSTCLICK'
                group by 1, 4
                union all
                select i.user_id, max(rownum) as max_row, min(rownum) as min_row, 1 as converter
                from {"tmp."+self.uniqueID+"imps"} i join {"tmp."+self.uniqueID+"activities"} a
                on i.user_id = a.user_id
                    and i.event_time > a.min_time
                    and i.event_time < a.max_time
                where a.event_sub_type = 'POSTVIEW'
                group by 1, 4)
            group by 1,2,3));"""


        if self.format == 'node':
            end_stmt = f"""
            select
                a.{self.eval_dimension} as start_node,
                b.{self.eval_dimension} as end_node,
                count(distinct a.user_id) as uniques,
                count(*) as count
            from {"tmp."+self.uniqueID+"imps"} a
            left join {"tmp."+self.uniqueID+"imps"} b
            on a.user_id = b.user_id
                and b.rownum - a.rownum = 1
            join {"tmp."+self.uniqueID+"exposed_nulls"} n
            on a.user_id = n.user_id
                and b.rownum <= n.max_row
            group by 1,2

            union all
            select
                a.{self.eval_dimension} as start_node,
                b.{self.eval_dimension} as end_node,
                count(distinct a.user_id) as uniques,
                count(*) as count
            from {"tmp."+self.uniqueID+"imps"} a
            left join {"tmp."+self.uniqueID+"imps"} b
            on a.user_id = b.user_id
                and b.rownum - a.rownum = 1
            join {"tmp."+self.uniqueID+"exposed_nulls2"} n2
            on a.user_id = n2.user_id
                and b.rownum <= n2.max_row
                and b.rownum > n2.min_row
            group by 1,2

	    union all
            select
                'start' as start_node,
                {self.eval_dimension} as end_node,
                count(distinct user_id) as uniques,
                count(*) as count
            from {"tmp."+self.uniqueID+"imps"}
            where rownum = 1
            group by 1, 2
            
            
            union all
            select
                {self.eval_dimension} as start_node,
                case when converter = 1 then 'conv' when converter = 0.4 then 'conv'
                else 'null' end as end_node,
                count(distinct i.user_id) as uniques,
                sum(converter) as count
            from {"tmp."+self.uniqueID+"imps"} i
            join {"tmp."+self.uniqueID+"exposed_nulls2"} n2
            on i.user_id = n2.user_id
               and i.rownum = n2.max_row
            group by 1, 2
            
            
            union all
            select
                {self.eval_dimension} as start_node,
                case when converter = 1 then 'conv' when converter = 0.4 then 'conv'
                else 'null' end as end_node,
                count(distinct i.user_id) as uniques,
                sum(converter) as count
            from {"tmp."+self.uniqueID+"imps"} i
            join {"tmp."+self.uniqueID+"exposed_nulls"} n
            on i.user_id = n.user_id
                and i.rownum = n.max_row
            group by 1, 2"""

        elif self.format == 'path':
            end_stmt = f""",
            paths as (
            select
                i.user_id,
                converter,
                string_agg({self.eval_dimension}, " > " order by rownum asc) as path
            from imps i
            join exposed_nulls n
            on i.user_id = n.user_id
                and i.rownum <= max_row
            group by 1, 2)
            select
                path,
                countif(converter > 0) as converters,
                count(*) as uniques
            from paths
            group by 1"""

        return window_stmts + end_stmt

    def create_adh_query(self, service):
        """Inserts new query into ADH

        Parameters
        ----------
        service : googleapiclient.discovery.Resource
            Google API resource for ADH

        Returns
        -------
        adh_query_name : str
            ADH-generated name of the inserted query

        Raises
        ------
        QueryExistsError
            If query already exists in ADH
        """

        if self.query_name is not None:
            msg = 'Query already exists in ADH. Delete query from ADH before recreating'
            raise QueryExistsError(msg)

        parameters = {}
        events = self.campaign.events
        for i in range(len(events)):
            parameters[f'event_start{i}'] = {'type': {'type': 'TIMESTAMP'}}
            parameters[f'event_end{i}'] = {'type': {'type': 'TIMESTAMP'}}
            parameters['activity_ids'] = {'type': {'arrayType': {'type': 'INT64'}}}
        if self.campaign.advertiserId:
            parameters['advertiser_Id'] = {'type': {'arrayType': {'type': 'INT64'}}} 
        if self.campaign.id:
            parameters['campaign_id'] = {'type': {'arrayType': {'type': 'INT64'}}}         
        if self.campaign.ids:
            parameters['search_campaign_id'] = {'type': {'arrayType': {'type': 'INT64'}}}     
        if self.campaign.country_code:
            parameters['country_code'] = {'type': {'arrayType': {'type': 'STRING'}}}
        return self.create(service, self.adh_text, parameter_types=parameters)

    def start_adh_query(self, service):
        """Starts a new run of the query in ADH

        Parameters
        ----------
        service : googleapiclient.discovery.Resource
            Built service object for accessing ADH

        Returns
        -------
        adh_operation_name : str
            ADH-generated name of the operation

        Raises
        ------
        TypeError
            If no ADH query is found
        """

        if self.query_name is None:
            # Look for query matching title in ADH
            response = service.customers().analysisQueries().list(
                parent=f'customers/{self.customer_id}',
                filter=f'title = {self.title}'
            ).execute()
            if 'queries' in response.keys():
                self.query_name = response['queries'][0]['name']
            else:
                msg = 'No ADH query found. Create a query before starting it.'
                raise TypeError(msg)

        values = {}
        events = self.campaign.events
        num_events = len(events)
        for i in range(num_events):
            values[f'event_start{i}'] = {'value': events[i]['start_timestamp']}
            values[f'event_end{i}'] = {'value': events[i]['end_timestamp']}
        values['activity_ids'] = {'arrayValue': {'values': []}}        
        for id in self.campaign.activityIds:
            values['activity_ids']['arrayValue']['values'].append(
                {'value': str(id)})
        if self.campaign.advertiserId:
            values['advertiser_Id'] = {'arrayValue': {'values': []}}
            for id in self.campaign.advertiserId:
                values['advertiser_Id']['arrayValue']['values'].append(
                    {'value': str(id)})
        if self.campaign.id:
            values['campaign_id'] = {'arrayValue': {'values': []}}
            for id in self.campaign.id:
                values['campaign_id']['arrayValue']['values'].append(
                    {'value': str(id)})
        if self.campaign.ids:
            values['search_campaign_id'] = {'arrayValue': {'values': []}}
            for id in self.campaign.ids:
                values['search_campaign_id']['arrayValue']['values'].append(
                    {'value': str(id)})
        if self.campaign.country_code:
            values['country_code'] = {'arrayValue': {'values':[]}}
            for code in self.campaign.country_code:
                values['country_code']['arrayValue']['values'].append(
                        {'value': str(code)})
        return self.start(service, parameter_values=values)

    def retrieve_adh_results(self, service, bq_client, overwrite=False):
        """Returns the results of ADH query as a Pandas dataframe

        Parameters
        ----------
        service : googleapiclient.discovery.Resource
            Built service object for accessing ADH
        bq_client : google.cloud.bigquery.Client
            Built BigQuery client object for accessing the BigQuery API
        overwrite : bool, optional
            If True, overwrites any previous query results with a new ADH run.
            Default is False

        Returns
        -------
        df : pandas.DataFrame
            Results of the query run as a Pandas DataFrame
        """

        if overwrite or not self.results_available(bq_client):
            self.result(service)
        df = self.to_dataframe(service, bq_client)
        return df

    @property
    def bq_imps_table(self):
        table = (f"`essence-dt-raw.{self.dcm_account_id}."
            f"impressions_{self.campaign.advertiserId}`")
        return table
    @property
    def bq_clicks_table(self):
        table = (f"`essence-dt-raw.{self.dcm_account_id}."
            f"clicks_{self.campaign.advertiserId}`")
        return table
    @property
    def bq_attr_text(self):
        """Returns sql string to be queried in BigQuery"""

        time_string = "%Y-%m-%d %H:%M:%S"

        def format_time(event, timestamp):
            new_datetime = datetime.strptime(event[timestamp], time_string)
            return new_datetime - self.timezone_offset

        event_times = [(
            format_time(event, 'start_timestamp'),
            format_time(event, 'end_timestamp')
        ) for event in self.campaign.events]
        partition_start = min(event[0] for event in event_times).date()
        partition_end = max(event[1] for event in event_times).date() + timedelta(days=1)

        event_stmts = " or ".join((f'timestamp_micros(event_time) between "{start.strftime(time_string)}" '
            f'and "{end.strftime(time_string)}"')
            for start, end in event_times)

        act_stmt_list = []
        for id in self.floodlight_config_ids:
            table = f"`essence-dt-raw.{self.dcm_account_id}.activity_{self.campaign.advertiserId}`"
            act_stmt = f"""select
                    user_id,
                    activity_id,
                    min(event_time) as min_time
                from {table}
                where _PARTITIONTIME >= "{str(partition_start)} 00:00:00"
                    and _PARTITIONTIME < "{str(partition_end)} 00:00:00"
                    and {event_stmts}
                group by 1, 2"""
            act_stmt_list.append(act_stmt)
        activities_stmts = "\nunion all\n".join(act_stmt_list)

        window_stmts = f"""
            with imp as (SELECT
                cast({self.eval_dimension} as string) as {self.eval_dimension},
                user_id,
                event_time,
                ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_time ASC) AS rownum
            FROM
                {self.bq_imps_table}
            WHERE
                _PARTITIONTIME >= "{str(self.campaign.startDate)} 00:00:00"
                AND _PARTITIONTIME < "{str(self.campaign.endDate + timedelta(days=1))} 00:00:00"
                AND campaign_id = {self.campaign.id}
                and user_id != '0'
            group by 1, 2, 3),
            clicks as (select 
                  cast({self.eval_dimension} as string) as {self.eval_dimension}, 
                  user_id, 
                  event_time, 
                  ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_time ASC) AS rownum 
            from {self.bq_clicks_table}
            WHERE 
                _PARTITIONTIME >= "{str(self.campaign.startDate)} 00:00:00"
                AND _PARTITIONTIME < "{str(self.campaign.endDate + timedelta(days=1))} 00:00:00"
                AND campaign_id = {self.campaign.ids}
                and user_id != '0'
            group by 1,2,3),            
            imps as (select 
                 user_id, 
                 placement_id, 
                 event_time, 
                 ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_time ASC) AS rownum 
                 from (select 
                       imp.user_id as user_id, 
                       imp.{self.eval_dimension} as {self.eval_dimension}, 
                       imp.event_time as event_time, 
                       imp.rownum as rownum from imp 
                       union all
                       select 
                       clicks.user_id as user_id, 
                       clicks.{self.eval_dimension} as {self.eval_dimension}, 
                       clicks.event_time as event_time, 
                       clicks.rownum as rownum from clicks)
            group by 1, 2, 3),
            activities as (
            select
                user_id,
                min(min_time) as min_time
            from (
                {activities_stmts}
            )
            where user_id != '0'
                and activity_id in UNNEST({self.campaign.activityIds})
            group by 1),
            exposed_nulls as (
            select
                user_id,
                min(max_row) as max_row,
                max(num) as converter
            from (
                select i.user_id, 1 as num, max(rownum) as max_row
                from imps i join activities a
                on i.user_id = a.user_id
                    and i.event_time < a.min_time
                group by 1, 2
                union all
                select user_id, 0 as num, max(rownum) as max_row
                from imps i group by 1, 2)
                group by 1)"""

        if self.format == 'node':
            end_stmt = f"""
            select
                a.{self.eval_dimension} as start_node,
                b.{self.eval_dimension} as end_node,
                count(distinct a.user_id) as uniques,
                count(*) as count
            from imps a
            left join imps b
            on a.user_id = b.user_id
                and b.rownum - a.rownum = 1
            join exposed_nulls n
            on a.user_id = n.user_id
                and b.rownum < n.max_row
            group by 1,2
            union all
            select
                {self.eval_dimension} as start_node,
                case when exposure = 1 then 'conv'
                else 'null' end as end_node,
                count(distinct i.user_id) as uniques,
                count(*) as count
            from imps i
            join exposed_nulls n
            on i.user_id = n.user_id
                and i.rownum = n.max_row
            group by 1, 2
            union all
            select
                'start' as start_node,
                {self.eval_dimension} as end_node,
                count(distinct user_id) as uniques,
                count(*) as count
            from imps
            where rownum = 1
            group by 1, 2"""

        elif self.format == 'path':
            end_stmt = f""",
            paths as (
            select
                i.user_id,
                converter,
                string_agg({self.eval_dimension}, " > " order by rownum asc) as path
            from imps i
            join exposed_nulls n
            on i.user_id = n.user_id
                and i.rownum <= max_row
            group by 1, 2)
            select
                path,
                countif(converter = 1) as converters,
                count(*) as uniques
            from paths
            group by 1
            having countif(converter = 1)>0"""

        return window_stmts + end_stmt

    def bq_query(self, bq_client):
        """Runs and returns the results of BigQuery query

        Parameters
        ----------
        bq_client : google.cloud.bigquery.Client
            Built BigQuery client object for accessing the BigQuery API

        Returns
        -------
        df : pandas.DataFrame
        """

        job = bq_client.query(self.bq_attr_text)
        result = job.result()
        df = result.to_dataframe()
        return df

    def bq_imps_cost(self, bq_client):
        """Returns reach, impressions, and spend by the eval_dimension from BigQuery DT Transfer data
        
        Parameters
        ----------
        bq_client : google.cloud.bigquery.Client
            Built BigQuery client object for accessing the BigQuery API
        
        Returns
        -------
        df : pandas.DataFrame
        """

        end_date = str(self.end_date + timedelta(days=1))
        sql = f"""select
            {self.eval_dimension} as channel,
            count(*) as impressions,
            count(distinct user_id) as uniques,
            sum(dbm_media_cost_usd) / 1000000000 as spend
        from {self.bq_imps_table}
        where
            _PARTITIONTIME >= "{str(self.start_date)} 00:00:00"
            AND _PARTITIONTIME < "{end_date} 00:00:00"
            AND campaign_id = {self.campaign.id}
            AND user_id != '0'
        group by 1"""

        job = bq_client.query(sql)
        result = job.result()
        df = result.to_dataframe()
        return df