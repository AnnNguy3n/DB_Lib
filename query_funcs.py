def get_list_table():
    return '''SELECT name FROM sqlite_master WHERE type = "table";'''


def create_table(len_formula, list_field, cycle):
    list_formula_col = [f'"E{i}" INTEGER NOT NULL,' for i in range(len_formula)]
    list_field_col = [f'"{field[0]}" {field[1]},' for field in list_field]
    temp = "\n    "
    return f'''CREATE TABLE "{cycle}_{len_formula}" (
    "id" INTEGER NOT NULL,
    {temp.join(list_formula_col)}
    {temp.join(list_field_col)}
    PRIMARY KEY ("id")
)'''


def create_checkpoint(len_formula):
    list_formula_col = [f'"E{i}" INTEGER NOT NULL,' for i in range(len_formula)]
    temp = "\n    "
    return f'''CREATE TABLE "checkpoint_{len_formula}" (
    "id" INTEGER NOT NULL,
    {temp.join(list_formula_col)[:-1]}
)'''


def get_last_row(table_name):
    return f'''SELECT * FROM "{table_name}" ORDER BY "id" DESC LIMIT 1;'''


def insert_rows(table_name, list_of_list_value, list_field, len_formula, mode, start_ID):
    list_field_name = [f_[0] for f_ in list_field]
    text_1 = f'''("id", {", ".join([f'"{field}"' for field in list_field_name])})'''
    if mode == 1:
        temp_text = ", ".join([f'"E{i}"' for i in range(len_formula)])
        text_1 = text_1[0:7] + temp_text + ", " + text_1[7:]
        if len(list_field) == 0:
            text_1 = text_1[:-3] + text_1[-1:]

    text_2 = ""
    for list_value in list_of_list_value:
        text = ""
        for value in list_value:
            if type(value) == str:
                text += f'"{value}",'
            else:
                text += f"{value},"

        text_2 += f"({start_ID},{text[:-1]}),"
        start_ID += 1

    return f'''INSERT INTO "{table_name}"{text_1} VALUES {text_2[:-1]};'''
