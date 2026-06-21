# causalbundle/causal/do_operator.py

def do_intervention(data, variable, value):
    data_copy = data.copy()
    data_copy[variable] = value
    return data_copy
