"""Oracle connection helper for raw academic data extraction.

Credential constants are intentionally expected from the execution environment
or an ignored local settings layer; this module only centralizes the connection
call used by notebooks and ingestion scripts.
"""

import oracledb

## here put your data 

# Fill these values with your Oracle server credentials before calling
# get_connection, or inject them from a local configuration module.


def get_connection():
    # Keep the connection construction in one function so extraction notebooks do
    # not duplicate credential names or Oracle connection parameters.
    return oracledb.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        service_name=DB_SERVICE_NAME,
    )
