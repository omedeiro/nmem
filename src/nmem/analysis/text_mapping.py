

BIT_TEXT_MAP_V1 = {
    "0": "WR0",
    "1": "WR1",
    "N": "",
    "R": "RD",
    "E": "Read \nEnable",
    "W": "Write \nEnable",
}

BIT_TEXT_MAP_V2 = {
    "0": "WR0",
    "1": "WR1",
    "N": "",
    "R": "RD",
    "E": "ER",
    "W": "EW",
    "z": "RD0",
    "Z": "W0R1",
    "o": "RD1",
    "O": "W1R0",
}

BIT_TEXT_MAP_V3 = {
    "0": "0",
    "1": "1",
    "N": "",
    "R": "",
    "E": "",
    "W": "",
}

def get_text_from_bit(bit: str, version: int = 1) -> str:
    if version == 1:
        return BIT_TEXT_MAP_V1.get(bit, None)
    elif version == 2:
        return BIT_TEXT_MAP_V2.get(bit, None)
    elif version == 3:
        return BIT_TEXT_MAP_V3.get(bit, None)
    else:
        return None