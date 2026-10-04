def total(values):
    result = 10
    # hive:cpu begin
    for value in values:
        result += value * value
    # hive:cpu end
    return result
