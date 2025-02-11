from pymongo import MongoClient
from config import Config
import os
import json
import pandas as pd

#client = MongoClient(Config.MONGO_URI)
#print("Databases:", client.list_database_names())

def get_heros():
    """ Returns M5 Heros used in pro esports as a list"""
    pth = os.path.join("mlbb_work","cleaned", "all_hero_data.json")
    with open(pth, 'r') as f:
        js = f.read()
    df_hero = pd.DataFrame.from_dict(json.loads(js))
    #First fix hero names
    df_hero['Hero'] = [hero.split(",")[0] for hero in df_hero['Hero']]
    df_hero_cols = ['icon', 'hero', 'id', 'roles', 'playstyie', 'lanes', 'birth', 'prices', 'releasedate']
    df_hero.columns = df_hero_cols
    df_hero.drop(columns=['icon','releasedate','prices','birth'], inplace=True)
    df_hero.drop(df_hero.index[-1])
    
    hero_arr = [x for x in df_hero['hero']]
    hero_arr.sort()
    #print(hero_arr)
    
    return hero_arr

get_heros()
"""insert some data X into mongodb
"""

"""
##########
db = client.get_database('m5')
    #print("Collections:", db.list_collection_names())
    m5_collections = db.get_collection('hero_winrates')
    b = [hero for hero in m5_collections.find()]
###########
json_file_path = os.path.join("app","backend","data_gather","mlbb_work", "Mlbb_json", "M5_uncut.json")
if os.path.exists(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        # Load the JSON data from the file
        json_data = json.load(json_file)
else:
    print("Could not open path")
    pass

try:
    #app\backend\data_gather\mlbb_work\Mlbb_json\M5_uncut.json
    m5_collections.insert_many(json_data)
except Exception as e:
    print(e)

print("Done")
"""