import os
import csv
import glob
import numpy as np
import collections


Prices = collections.namedtuple('Prices', field_names=['DES', 'open', 'high', 'low', 'close', 'volume', 'sent'])

SENT_NORMALIZE = 100.0


def read_csv(file_name, sep=',', filter_data=True, fix_open_price=False):
    print("Reading", file_name)
    with open(file_name, 'rt', encoding='utf-8') as fd:
        reader = csv.reader(fd, delimiter=sep)
        h = next(reader)
        if '<OPEN>' not in h and sep == ',':
            return read_csv(file_name, ';')
        if '<VOL>' in h:
            vol_key = '<VOL>'
        elif '<VOLUME>' in h:
            vol_key = '<VOLUME>'
        else:
            vol_key = None
        sent_key = '<SENT>' if '<SENT>' in h else None
        base_cols = ('<DES>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>')
        extra_cols = []
        if vol_key is not None:
            extra_cols.append(vol_key)
        if sent_key is not None:
            extra_cols.append(sent_key)
        indices = [h.index(s) for s in (*base_cols, *extra_cols)]
        d, o, h, l, c, v, s = [], [], [], [], [], [], []
        count_out = 0
        count_filter = 0
        count_fixed = 0
        prev_vals = None
        for row in reader:
            raw_vals = [row[idx] for idx in indices]
            base_floats = list(map(float, raw_vals[:5]))
            pd, po, ph, pl, pc = base_floats
            offset = 5
            if vol_key is not None:
                pv = float(raw_vals[offset])
                offset += 1
            else:
                pv = 0.0
            if sent_key is not None:
                sent_str = raw_vals[offset].strip()
                ps = float(sent_str) if sent_str != '' else 0.0
                offset += 1
            else:
                ps = 0.0

            if filter_data and all(map(lambda x: abs(x - base_floats[0]) < 1e-8, base_floats)):
                count_filter += 1
                continue

            if filter_data and po == 0 and ph == 0 and pl == 0 and pc == 0 and pv == 0:
                count_filter += 1
                continue

            # fix open price for current bar to match close price for the previous bar
            if fix_open_price and prev_vals is not None:
                _, _, _, _, ppc, _, _ = prev_vals
                if abs(po - ppc) > 1e-8:
                    count_fixed += 1
                    po = ppc
                    pl = min(pl, po)
                    ph = max(ph, po)
            count_out += 1
            d.append(pd)
            o.append(po)
            c.append(pc)
            h.append(ph)
            l.append(pl)
            v.append(pv)
            s.append(ps)
            prev_vals = (pd, po, ph, pl, pc, pv, ps)
    print("Read done, got %d rows, %d filtered, %d open prices adjusted" % (
        count_filter + count_out, count_filter, count_fixed))
    return Prices(DES=np.array(d, dtype=np.float32),
                  open=np.array(o, dtype=np.float32),
                  high=np.array(h, dtype=np.float32),
                  low=np.array(l, dtype=np.float32),
                  close=np.array(c, dtype=np.float32),
                  volume=np.array(v, dtype=np.float32),
                  sent=np.array(s, dtype=np.float32))


def prices_to_relative(prices):
    """
    Convert prices to relative in respect to open price
    :param ochl: tuple with open, close, high, low
    :return: tuple with open, rel_close, rel_high, rel_low
    """
    assert isinstance(prices, Prices)
    rh = (prices.high - prices.open) / prices.open
    rl = (prices.low - prices.open) / prices.open
    rc = (prices.close - prices.open) / prices.open
    # Normalize sentiment (raw 0-100 scale) to ~0-1 to match other features.
    sent_norm = prices.sent / SENT_NORMALIZE
    return Prices(DES=prices.DES, open=prices.open, high=rh, low=rl, close=rc,
                  volume=prices.volume, sent=sent_norm)


def load_relative(csv_file):
    return prices_to_relative(read_csv(csv_file))


def price_files(dir_name):
    result = []
    for path in glob.glob(os.path.join(dir_name, "*.csv")):
        result.append(path)
    return result


def load_year_data(year, basedir='data'):
    y = str(year)[-2:]
    result = {}
    for path in glob.glob(os.path.join(basedir, "*_%s*.csv" % y)):
        result[path] = load_relative(path)
    return result
