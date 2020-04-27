# Markov Chain Attribution for Python

This project performs Markov chain attribution on a given dataset. It accepts a Pandas DataFrame. The file `markov_attribution_script.py` is intended to be run as a script, which will query DCM data from BigQuery or ADH and perform the attribution methodology on queried data. Without running the script, `attribution.py` can be imported into other code to analyze non-DCM data.

## Getting Started

Good luck on your journey to slightly better than linear attribution methodology!

### Script Prerequisites

In order to run `markov_attribution_script.py` and `query_to_csv.py`, you will need either OAuth client secrets or service account credentials from Google with the BigQuery API enabled. To use ADH as your data source, you will also need to enable the ADH API as well as have an ADH developer key.

You must either pass a config file path or name your config file "config.json." The config file will define certain attributes about your API credentials, DCM campaign, and data source.

The script, `attribution_from_file.py` will take a csv of conversion data, perform attribution methodologies of your choice, and output the results as csvs.

## Config File Format

The config file defines authorization, campaign, and data source information. It is required to run `markov_attribution_script.py` and `query_to_csv.py`.

### JSON Representation

The Current structure is hard-coded when using a Site_Tactic eval_dimension. Contact maximilian.schmitt@ for further details.

```json
{
    "auth": {
        "developer_key": string,
        "service_key_path": string,
        "client_secret_path": string
    },
    "campaign": {
        "id": Array[int],
	"ids": Array[int],
	"country_code": Array[string],
        "advertiser_id": Array[inti,
        "start_date": string,
        "end_date": string,
        "activity_ids": Array[int],
        "events": [
            {
                "start_timestamp": string,
                "end_timestamp": string
            }
        ]
    },
    "source": {
        "use": string,
        "eval_dimension": string,
        "time_zone": string,
        "query_format": string,
        "ADH": {
            "customer_id": int,
            "data_customer_id": int,
            "overwrite": bool
        },
        "BQ": {
            "dcm_account_id": int,
            "floodlight_config_ids": Array[int]
        }
    }
}
```

### Fields

| Field | Data Type | Description |
|-------|-----------|-------------|
| developer_key | string | API key with ADH API enabled |
| service_key_path | string | Path to service key file |
| client_secret_path | string | Path to client secret file |
| id | Array[int] | List DCM Campaign ID(s) |
| ids | Array[int] | List DCM Search Campaign ID(s)
| country_code | Array[string] | List of DCM Country_Codes
| advertiser_id | Array[int] | DCM Advertiser ID(s) - needed for BigQuery only
| start_date | string | Start date of the campaign ex. "2019-04-01" |
| end_date | string | End date of the campaign ex. "2019-04-29" |
| activity_ids | Array[int] | List of activity ID(s) that will be counted as a conversion ex. [1234, 5678]
| events | Array[{"start_timestamp": string (Timestamp format), "end_timestamp": string (Timestamp format)}] | List of time intervals during which a conversion could happen. Values should be in the following format: "yyyy-mm-dd hh:mm:ss" |
| use | string | Specifies which data source to use. "BQ" or "ADH" |
| eval_dimension | string | Dimension to evaluate conversions on ex. placement_id |
| time_zone | string | Time zone to evaluate dates & times on ex. "US/Eastern" |
| query_format | string | Format the query results DataFrame will be in. "path" ( path='channel_1 > channel_2 > channel_3', converters=2, uniques=4 ) or "node" (start_node=channel_1, end_node=channel_2, count=4) |
| customer_id | int | ADH Customer ID |
| data_customer_id | int | ADH Customer ID of data location, if different from customer_id |
| overwrite | bool | If false, checks to see if results already exist in BQ and returns those instead of creating a new query |
| dcm_account_id | int | DCM account ID of advertiser |
| floodlight_config_id | Array[int] | List of floodlight configuration ID(s) of activity IDs |
