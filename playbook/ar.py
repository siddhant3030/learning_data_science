import requests
from bigquery_schema_generator.generate_schema import SchemaGenerator
from gbqschema_converter.gbqschema_to_jsonschema import json_representation as converter




username = "shri_dashboard"
password = "p4shri_dashboard"
    
url = f"https://kf.kobotoolbox.org/api/v2/assets/aLKzTfJAw5KKbtRxLUncwh/data.json"
response = requests.get(url, auth=(username, password))
response_json = response.json()
result = response_json.get('results')

generator = SchemaGenerator(input_format="dict", infer_mode="NULLABLE", preserve_input_sort_order="true")

schema_map, error_logs = generator.deduce_schema(input_data=result)
schema = generator.flatten_schema(schema_map)
schema_json = converter(schema)
schema = schema_json["definitions"]["element"]["properties"]
print(schema)

