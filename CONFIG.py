DB_PATH = "/media/nguyen/VIS_DB/DB"

DATA_PATH = "/home/nguyen/Desktop/DB_Lib/HOSE_File3_2023_Field.xlsx"

LABEL = "VN_Y"
'''
Label, vi du: "VN_Y" - data Viet Nam, chu ky nam
              "VN_Q" - data Viet Nam, chu ky quy
              "JP_Y" - data Nhat Ban, chu ky nam
'''

INTEREST = 1.06 # Lai suat khi ko dau tu trong 1 chu ky

NUM_CYCLE = 14 # Sinh cong thuc cho bao nhieu chu ky, tinh tu chu ky cuoi cung
MIN_CYCLE = 2007 # Su dung data dau vao tu chu ky bao nhieu

METHOD = 1 # Phuong phap sinh

DB_FIELDS = {
    "get_inv_max_infor": [
        ("ProMax", "REAL"),
        ("GeoMax", "REAL"),
        ("HarMax", "REAL")
    ],
    "get_inv_ngn_infor": [
        ("Nguong", "REAL"),
        ("ProNgn", "REAL"),
        ("GeoNgn", "REAL"),
        ("HarNgn", "REAL")
    ],
    "get_tf_score": [
        ["TrFScr", "REAL"]
    ],
    "get_ac_score": [
        ["AccScr", "REAL"]
    ],
    "get_inv_ngn2_infor" : [
        ("Nguong2", "REAL"),
        ("ProNgn2", "REAL"),
        ("GeoNgn2", "REAL"),
        ("HarNgn2", "REAL")
    ],
    "get_inv_ngn1_2_infor": [
        ("Nguong1_2", "REAL"),
        ("ProNgn1_2", "REAL"),
        ("GeoNgn1_2", "REAL"),
        ("HarNgn1_2", "REAL")
    ],
    "get_inv_ngn1_3_infor": [
        ("Nguong1_3", "REAL"),
        ("ProNgn1_3", "REAL"),
        ("GeoNgn1_3", "REAL"),
        ("HarNgn1_3", "REAL")
    ]
}

'''
DB_FIELDS
    Key: Ten ham sinh du lieu co trong file "get_value_funcs.py"
    Value: List cac du lieu dau ra cua ham, co dang tuple(ten du lieu, kieu du lieu)
'''

MODE = "generate" # or "update"
NUM_FUNC = 2 # Chi kha dung khi MODE = "update"

NUM_CHILD_PROCESS = 9


# ===========================================================================
import os
import pandas as pd
import json
import sqlite3
from Methods.base import Base
import get_value_funcs as gvf


def check_data_operands(op_name_1: dict, op_name_2: dict):
    if len(op_name_1) != len(op_name_2): return False

    op_1_keys = list(op_name_1.keys())
    op_2_keys = list(op_name_2.keys())
    for i in range(len(op_name_1)):
        if op_name_1[op_1_keys[i]] != op_name_2[op_2_keys[i]]:
            return False

    return True


def check_config():
    folder_data = f"{DB_PATH}/{LABEL}"
    os.makedirs(folder_data, exist_ok=True)

    data = pd.read_excel(DATA_PATH)
    data = data[data["TIME"] >= MIN_CYCLE]
    base = Base(data)

    if not os.path.exists(folder_data + "/operand_names.json"):
        with open(folder_data + "/operand_names.json", "w") as fp:
            json.dump(base.operand_name, fp, indent=4)
        operand_name = base.operand_name
    else:
        with open(folder_data + "/operand_names.json", "r") as fp:
            operand_name = json.load(fp)

    if not check_data_operands(base.operand_name, operand_name):
        raise Exception("Sai data operands, kiem tra lai ten truong, thu tu cac truong trong data")

    folder_method = folder_data + f"/METHOD_{METHOD}"
    os.makedirs(folder_method, exist_ok=True)
    connection = sqlite3.connect(f"{folder_method}/f.db")

    if MODE == "generate":
        list_gvf = [getattr(gvf, key) for key in DB_FIELDS.keys()]
        list_field = []
        for key in DB_FIELDS.keys():
            list_field += DB_FIELDS[key]
    elif MODE == "update":
        ...
    else: raise

    return data, connection, list_gvf, list_field
