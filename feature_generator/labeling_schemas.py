TOW_IDS_ATTACK_LABELS = ["C_D", "C_R", "M_F", "F_I", "P_I"]
NORMAL_KEY = "Normal"
ABNORMAL_KEY = "Abnormal"
INDEX_COLUMN = "index"
DESCRIPTION_COLUMN = "Description"
CLASS_COLUMN = "Class"

def avtp_intrusion_labeling_schema(y_sequence):
    # Expects y_sequence as a list
    # Labeling schema: if there is an attack in the sequence, the sequence is considered as an attack
    seq_y = 0

    if 1 in y_sequence:
        seq_y = 1

    return seq_y

def tow_ids_one_class_labeling_schema(y_sequence):
    # Expects y_sequence as a dataframe containing ["Class", "Description"] columns
    # Labeling schema: if there is an attack in the sequence, the sequence is considered as an attack
    seq_y = 0

    if ABNORMAL_KEY in y_sequence[CLASS_COLUMN].values:
        seq_y = 1

    return seq_y

def tow_ids_multi_class_labeling_schema(y_sequence):
    # Expects y_sequence as a dataframe containing ["Class", "Description"] columns
    # Labeling schema: if there are multiple attacks in the sequence, the label will correspond to the most frequent attack
    seq_y = NORMAL_KEY

    indexes = y_sequence[DESCRIPTION_COLUMN].value_counts().sort_values(ascending=False).reset_index()
    indexes_list = list(indexes[INDEX_COLUMN].values)

    set_attacks = set(TOW_IDS_ATTACK_LABELS)
    set_sequence_indexes = set(indexes_list)

    intersect = any(set_atk in set_sequence_indexes for set_atk in set_attacks)

    if intersect is True:
        attacks_mask = indexes[INDEX_COLUMN].isin(TOW_IDS_ATTACK_LABELS)
        indexes_attacks = indexes[attacks_mask]
        seq_y = indexes_attacks[INDEX_COLUMN].values[0]

    return seq_y
