import functools
import inspect

def validate_config(func):
    """Checks to make sure config file is valid."""

    @functools.wraps(func)
    def wrapper(config):
        bad_values = [None, "", []]

        if (config['auth']['service_key_path'] in bad_values) and (config['auth']['client_secret_path'] in bad_values):
            raise TypeError("service_key_path or client_secret_path required.")

        for key, value in config['campaign'].items():
            if key == 'advertiser_id' and config['source']['use'] not in ['ALL','BQ']:
                continue
            if value in bad_values:
                raise TypeError(f"{key} must have a value.")

        if config['source']['use'] not in ['ADH','BQ','ALL']:
            raise ValueError("Parameter 'use' must be ADH, BQ, or ALL")
        if config['source']['eval_dimension'] in bad_values:
            raise TypeError("eval_dimension must have a value.")

        use = config['source']['use']
        if use in ['ALL','ADH']:
            if config['auth']['developer_key'] in bad_values:
                raise TypeError("Running query in ADH requires developer key.")
            if config['source']['ADH']['customer_id'] in bad_values:
                raise TypeError("A customer_id is required to run query in ADH.")

        if use in ['ALL','BQ']:
            for key, value in config['source']['BQ'].items():
                if value in [None, "", []]:
                    raise TypeError(f"{key} is required to run query in BigQuery.")
        
        res = func(config)
        return res
    return wrapper
