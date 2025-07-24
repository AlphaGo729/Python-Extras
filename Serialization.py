saveCode = ""
dev_idx = 0

def dev_WriteVal(text,delim,escape):
    for i in range(len(text)):
        c = text[i]
        if c == delim or c == escape:
            saveCode += escape
        saveCode += c
    saveCode += delim

def dev_ReadVal(delim, escape):
    value = ""
    while True:
        c = saveCode[dev_idx]
        dev_idx += 1
        if c == delim:
            break
        if c == "":
            return False
        if c == escape:
            c = saveCode[dev_idx]
            dev_idx += 1
        value += c
    return value

def Encode(list,delim="|", escape="\\"):
    saveCode = ""
    for i in range(len(list)):
        dev_WriteVal(list[i], delim, escape)
    return saveCode

def Decode(text,delim="|", escape="\\"):
    saveCode = text
    dev_idx = 0
    return_list = []
    while True:
        readval = dev_ReadVal(delim, escape)
        if readval == None:
            break
        return_list.append(dev_ReadVal(delim, escape))

    return return_list


