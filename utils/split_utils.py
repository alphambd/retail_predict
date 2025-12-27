def split_data(df, val_start, test_start=None):
    if test_start:
        train = df[df['ds'] < val_start]
        val = df[(df['ds'] >= val_start) & (df['ds'] < test_start)]
        test = df[df['ds'] >= test_start]
    else:
        train = df[df['ds'] <= val_start]
        val = df[df['ds'] > val_start]
        test = None
    return (train, val, test)