import pandas as pd
import numpy as np
import os
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def GSS_import_from_file(file_name):
    print('Importing the data using GSS library!')
    
    if not os.path.isfile(file_name):
        raise NameError('The file path is not a valid path! Please verify it.')
    file_format = file_name.split('.')[-1]
    
    if file_format=='xlsx':
        data= pd.read_excel(file_name,engine='openpyxl')
    elif file_format=='csv':
        data= pd.read_csv(file_name)
    elif file_format=='json':
        data = pd.read_json(file_name, orient ='split', compression = 'infer')
    else:
        raise ImportError('Importing the data in {} format is not supported. Choose xlsx, json and csv data only.'.format(file_format))

    col = data.columns

    print('Successfully imported {} number of data'.format(len(new_data)))
    return data

def GSS_import_from_query(_RunID, user_id=None):
    from queryrunner_client import Client as QueryrunnerClient
    from querybuilder_client import QuerybuilderClient
    from queryrunner_client import Client
    
    if user_id==None:
        user_id = input('Enter your Uber ID: ')
    
    qr = Client(user_email=user_id)
    cursor = qr.execute_run(_RunID)
    return cursor.to_pandas()


def GSS_image_from_url(url):
    from PIL import Image
    import urllib.request
    import io    

    image_data = urllib.request.urlopen(url).read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    return image

