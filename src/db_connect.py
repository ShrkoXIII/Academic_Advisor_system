import oracledb


# Fill these values with your Oracle server credentials.
DB_USER = "x"
DB_PASSWORD = "y"
DB_HOST = "z"
DB_PORT = 1521
DB_SERVICE_NAME = "RAS"


def get_connection():
    return oracledb.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        service_name=DB_SERVICE_NAME,
    )
