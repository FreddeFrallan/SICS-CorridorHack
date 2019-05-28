def get_class_names(dataset_name):
    if dataset_name == "fashionista_v1":
        return get_fashionista_v1_class_names()
    if dataset_name == "CFPD":
        return get_cfpd_class_names()
    if dataset_name == "modanet":
        return get_modanet_class_names()

def get_color_code(dataset_name):
    if dataset_name == "fashionista_v1":
        color_code = [[255,255,255],[226,196,196],[64,32,32],[255,0,0],[255,70,0],[255,139,0],[255,209,0],[232,255,0],[162,255,0],[93,255,0],[23,255,0],[0,255,46],[0,255,116],[0,255,185],[0,255,255],[0,185,255],[0,116,255],[0,46,255],[23,0,255],[93,0,255],[162,0,255],[232,0,255],[255,0,209],[255,0,139],[255,0,70]]
    if dataset_name == "CFPD":
        color_code = [[255,255,255],[255,67,0],[255,133,0],[255,200,0],[244,255,0],[177,255,0],[111,255,0],[44,255,0],[226,196,196],[64,32,32],[0,255,155],[0,255,222],[0,222,255],[0,155,255],[0,89,255],[0,22,255],[44,0,255],[206,176,176],[177,0,255],[244,0,255],[255,0,200],[255,0,133],[255,0,67]]
    if dataset_name == "modanet":
        color_code = [
            [255,255,255], # background
            [255,67,0], # bag
            [255,133,0], # belt
            [255,200,0], # boots
            [244,255,0], # footwear
            [177,255,0], # outer
            [111,255,0], # dress
            [44,255,0], # sunglasses
            [244,0,255],
            [255,0,0], # top
            [0,255,155], # shorts
            [0,255,222], # skirt
            [0,222,255], # headwear
            [0,155,255], # scart & tie
        ]
    return color_code

def get_modanet_class_names():
    return [
        "bg",
        "bag",
        "belt",
        "boots",
        "footwear",
        "outer",
        "dress",
        "sunglasses",
        "pants",
        "top",
        "shorts",
        "skirt",
        "headwear",
        "scarf & tie",
    ]

def get_fashionista_v1_class_names():
    return  [
        "background",
        "skin",
        "hair",
        "bag",
        "belt",
        "boots",
        "coat",
        "dress",
        "glasses",
        "gloves",
        "hat/headband",
        "jacket/blazer",
        "necklace",
        "pants/jeans",
        "scarf/tie",
        "shrit/blouse",
        "shoes",
        "shorts",
        "skirt",
        "socks",
        "sweater/cardigan",
        "tights/leggings",
        "top/t-shirt",
        "vest",
        "watch/bracelet"
    ]

def get_cfpd_class_names():
    return [
        "background",
        "t-shirt",
        "bag",
        "belt",
        "blazer",
        "blouse",
        "coat",
        "dress",
        "face",
        "hair",
        "hat",
        "jeans",
        "legging",
        "pants",
        "scarf",
        "shoe",
        "shorts",
        "skin",
        "skirt",
        "socks",
        "stocking",
        "sunglass",
        "sweater"
    ]

def get_fashionista_v1_class_weights():
    return [
            1.289868732318736, # "background"
            25.829792560370333, # "skin"
            47.77294484074959, # "hair"
            88.31674409901073, # "bag"
	    852.2918736769344, # "belt"
            163.58887184377463, # "boots"
            85.14414254220522, # "coat"
            39.64294024903837, # "dress"
            2572.5686565996402, # "glasses"
            8423.569598633647, # "gloves"
            570.7804832885844, # "hat/headband"
            68.24635106188666, # "jacket/blazer"
            1833.4913288351083, # "necklace"
            63.44897676946679, # "pants/jeans"
            465.5421413805797, # "scarf/tie"
            101.44495294904098, # "shrit/blouse"
            142.40136280298546, # "shoes"
            223.11189522969397, # "shorts"
            88.31421380742238, # "skirt"
            376.043612519538, # "socks"
            109.98408887927506, # "sweater/cardigan"
            74.39693904564656, # "tights/leggings"
            78.27838345297822, # "top/t-shirt"
            331.24124799774336, # "vest"
            2161.7357001972387 # "watch/bracelet
    ]
