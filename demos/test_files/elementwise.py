def transform(values):
    result = [0.0] * len(values)
    # hive:parallel begin
    for i in range(len(values)):
        scaled = values[i] * 2
        result[i] = scaled + 1
    # hive:parallel end
    return result
