def read_data(path):

    with open(path) as data:
        result = []
        for line in data:
            if not line.startswith('#'):
                line = line[:-1]
                result.append(line)
            
    return result