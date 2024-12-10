import numpy as np

cation_name_dict = {
    'Li',
    'Na',
    'K',
    'Zn',
    'Mg',
    'Fe',
    'Ca',
}

anion_name_dict = {
    'FSI',
    'TFSI',
    'BETI',
    'BNTI',
    'FSI-TFSI',
    'FSI-BETI',
    'FSI-BNTI',
    'TFSI-BETI',
    'TFSI-BNTI',
    'BETI-BNTI',
    'BF4',
    'OTF',
    'NO3',
    'SO4',
    'Cl',
    '2FCO3',
    '3FCO3',
    '4FCO3',
    '4FSO3',
    'Acetate',
    'TFA',
}

solvent_name_dict = {
    'H2O',
    'MeOH',
    'PC',
    'DME',
    'DEC',
    'ACN',
    'THF',
}

imide_anion_single_side_chain_length = {
    'FSI':  0,
    'TFSI': 1,
    'BETI': 2,
    'BNTI': 4,
}

# cation_radius_dict = {
#     'Li': [0.9 ],
#     'Na': [1.16],
#     'K':  [1.52],
#     'Zn': [-1, 0.88],
#     'Mg': [-1, 0.86],
#     'Fe': [-1, (0.75 + 0.92)/2, (0.69 + 0.785)/2],
#     'Ca': [-1, -1, 1.14],
# }

cation_radius_dict = {
    'Li': {1: 2.38},
    'Na': {1: 1.84},
    'K':  {1: 1.52},
    'Zn': {2: 1.745},
    'Mg': {2: 1.737},
    'Fe': {2: 1.706, 3: 1.355},
    'Ca': {2: 1.548},
}

feature_units_dict = {
    "concentration": "mol/kg",
    "cation valence": "",
    "cation size": r"$\mathrm{\AA}$",
    "anion valence": "",
    "anion mass": "g/mol",
    "FC chain avg": "",
    "FC chain asym": "",
    "solvent mass": "g/mol",
    "solvent density": r"$\mathrm{g/cm^3}$",
    "solvent boiling point": "K",
    "solvent dielectric": "",
    "solvent dipole": "D",
    "solvent viscosity": "cP",
}

def read_const_dict(filename: str, t: type = float) -> dict:
    a_map = {}
    with open(filename, 'r') as fin:
        for aline in fin:
            linelist = aline.strip().split()
            a_map[linelist[0]] = t(linelist[1])
    return a_map
const_map_path = "data"
atomic_mass_dict        = read_const_dict(f"{const_map_path}/atomic_mass_map.txt")
anion_valence_dict      = read_const_dict(f"{const_map_path}/anion_valence_map.txt")
solvent_density_dict    = read_const_dict(f"{const_map_path}/solvent_density_map.txt")
solvent_bp_dict         = read_const_dict(f"{const_map_path}/solvent_bp_map.txt")
solvent_dielectric_dict = read_const_dict(f"{const_map_path}/solvent_dielectric_map.txt")
solvent_dipole_dict     = read_const_dict(f"{const_map_path}/solvent_dipole_map.txt")
solvent_viscosity_dict  = read_const_dict(f"{const_map_path}/solvent_viscosity_map.txt")

def extract_component_number(x):
    if "(" in x and ")" in x:
        return x[x.index("(") + 1 : x.index(")")], float(x[x.index(")") + 1 : ])
    return x, 1

def reorder_imide_anion_side_chains(x):
    if "-" in x:
        x_list = x.split('-')
        assert len(x_list) == 2, "the number of side chains should be only 2"
        if imide_anion_single_side_chain_length[x_list[0]] < imide_anion_single_side_chain_length[x_list[1]]:
            return x
        else:
            return x_list[1] + "-" + x_list[0]
    return x

def smart_system_identification(chem_name: list, only_imide_salt: bool) -> tuple[list, list, bool]:
    component_num_list = []
    component_name_list = []
    component_dict = (cation_name_dict, anion_name_dict, solvent_name_dict)
    for i, a_component in enumerate(chem_name):
        num = []
        nam = []
        if '_' in a_component:
            a_component_list = a_component.strip().split('_')
        else:
            a_component_list = [a_component]

        for x in a_component_list:
            nam_, num_ = extract_component_number(x)
            if i == 0:
                nam_ = nam_.capitalize()
            if i == 1:
                nam_ = reorder_imide_anion_side_chains(nam_)
            num.append(num_)
            nam.append(nam_)

        assert all([v in component_dict[i] for v in nam]), f"Cannot find one of the component in component dictionary: {nam}."

        if i == 2:
            sum_num = sum(num)
            num = [v/sum_num for v in num]
            d = np.array([solvent_density_dict[v] for v in nam])
            d_sort_arg = np.argsort(d)
            num = [num[v] for v in d_sort_arg]
            nam = [nam[v] for v in d_sort_arg]
        component_num_list.append(num)
        component_name_list.append(nam)

    allow = True
    if any([len(v) != 1 for v in component_num_list]):
        allow = False
        print("warning: excluding any system that is not single salt with single solvent")
    if only_imide_salt:
        if len(component_num_list[1]) != 1 or ('FSI' not in component_name_list[1][0] and 'BETI' not in component_name_list[1][0] and 'BNTI' not in component_name_list[1][0]):
            allow = False
            print("warning: excluding any system that is not imide based single salt")

    return component_num_list, component_name_list, allow

def get_non_empty_index(x, val_tol = None):
    
    nonzero_index = np.logical_or(np.isnan(x), np.isinf(x))

    if val_tol is not None:
        nonzero_index = np.logical_or(nonzero_index, (x <= val_tol))

    nonzero_index = np.logical_not(nonzero_index)
    return nonzero_index

def is_pareto(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

################################################################################################

# 1A: <cation>
# 1B: <anion>
# 1C: <solvent>

# for bi-salt: <anion> = (<anion1>)N1_(<anion2>)N2
# for mixture solvent: <solvent> = (<solvent1>)V1_(<solvent2>)V2

#                               excel cell           
# example               col_1     col_2         col_3
# Zn(3FCO3)2-H2O(BS) -> Zn        (3FCO3)2     H2O
# Zn(4FSO3)2-H2O-BS  -> Zn        (4FSO3)2     H2O
# LITFSI-BETI H2O    -> Li        TFSI-BETI    H2O
# FeCl2-H2O          -> Fe        (Cl)2        H2O
# Fe(TFSI)2-H2O      -> Fe        (TFSI)2      H2O
# LiFSI-H2O-ACN      -> Li        FSI          H2O_ACN
# LiFSI-H2O-ACN      -> Li        FSI          (H2O)1_(ACN)1
# LITFSI-70M30H(V%)  -> Li        TFSI         (MeOH)70_(H2O)30
# LiBF4 in H2O       -> Li        BF4          H2O
# H2O-ACN-6(8.3:91.7)-> Zn	      (TFSI)2	   (H2O)83_(ACN)917

