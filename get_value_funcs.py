import numpy as np
import numba as nb


@nb.njit
def geomean(arr):
    log_sum = 0.0
    for num in arr:
        if num <= 0.0: return 0.0
        log_sum += np.log(num)

    return np.exp(log_sum/len(arr))


@nb.njit
def harmean(arr):
    deno = 0.0
    for num in arr:
        if num <= 0.0: return 0.0
        deno += 1.0/num

    return len(arr)/deno


@nb.njit
def _get_inv_max_infor(WEIGHT, INDEX, PROFIT, interest):
    '''
    Output: ProMax, GeoMax, HarMax
    '''
    size = INDEX.shape[0] - 1
    arr_profit = np.zeros(size)
    for i in range(size-1, -1, -1):
        start, end = INDEX[i], INDEX[i+1]
        temp = WEIGHT[start:end]
        max_ = np.max(temp)
        arr_idx_max = np.where(temp == max_)[0]
        if arr_idx_max.shape[0] == 1:
            arr_profit[size-i-1] = PROFIT[start:end][arr_idx_max[0]]
            if arr_profit[size-i-1] <= 0.0:
                break
        else:
            arr_profit[size-i-1] = interest

    GeoMax = geomean(arr_profit[:-1])
    HarMax = harmean(arr_profit[:-1])
    return arr_profit[-1], GeoMax, HarMax


@nb.njit
def get_inv_max_infor(WEIGHT, INDEX, PROFIT, SYMBOL, interest, num_cycle):
    '''
    Output: ProMax, GeoMax, HarMax
    '''
    result = []
    for i in range(num_cycle):
        ii = num_cycle - 1 - i
        WEIGHT_ = WEIGHT[INDEX[ii]:]
        PROFIT_ = PROFIT[INDEX[ii]:]
        INDEX_ = INDEX[ii:] - INDEX[ii]
        ProMax, GeoMax, HarMax = _get_inv_max_infor(WEIGHT_, INDEX_, PROFIT_, interest)
        result.append([ProMax, GeoMax, HarMax])

    return result


@nb.njit
def _get_inv_ngn_infor(WEIGHT, INDEX, PROFIT, interest):
    '''
    Output: Nguong, ProNgn, GeoNgn, HarNgn
    '''
    size = INDEX.shape[0] - 1
    arr_profit = np.zeros(size-1)
    temp_profit = np.zeros(size-1)
    max_profit = -1.0

    list_loop = np.zeros((size-1)*5)
    for k in range(size-1, 0, -1):
        start, end = INDEX[k], INDEX[k+1]
        temp_weight = WEIGHT[start:end].copy()
        temp_weight[::-1].sort()
        list_loop[5*(k-1):5*k] = temp_weight[:5]

    list_loop = np.unique(list_loop)
    for v in list_loop:
        C = WEIGHT > v
        temp_profit[:] = 0.0
        for i in range(size-1, 0, -1):
            start, end = INDEX[i], INDEX[i+1]
            if np.count_nonzero(C[start:end]) == 0:
                temp_profit[i-1] = interest
            else:
                temp_profit[i-1] = PROFIT[start:end][C[start:end]].mean()

        new_profit = geomean(temp_profit)
        if new_profit > max_profit:
            Nguong = v
            max_profit = new_profit
            arr_profit[:] = temp_profit[:]

    HarNgn = harmean(arr_profit)

    start, end = INDEX[0], INDEX[1]
    mask = WEIGHT[start:end] > Nguong
    if np.count_nonzero(mask) == 0:
        ProNgn = interest
    else:
        ProNgn = PROFIT[start:end][mask].mean()

    return Nguong, ProNgn, max_profit, HarNgn


@nb.njit
def get_inv_ngn_infor(WEIGHT, INDEX, PROFIT, SYMBOL, interest, num_cycle):
    '''
    Output: Nguong, ProNgn, GeoNgn, HarNgn
    '''
    result = []
    for i in range(num_cycle):
        ii = num_cycle - 1 - i
        WEIGHT_ = WEIGHT[INDEX[ii]:]
        PROFIT_ = PROFIT[INDEX[ii]:]
        INDEX_ = INDEX[ii:] - INDEX[ii]
        Nguong, ProNgn, GeoNgn, HarNgn = _get_inv_ngn_infor(WEIGHT_, INDEX_, PROFIT_, interest)
        result.append([Nguong, ProNgn, GeoNgn, HarNgn])

    return result


@nb.njit
def countTrueFalse(a, b):
    countTrue = 0
    countFalse = 0
    len_ = len(a)
    for i in range(len_ - 1):
        for j in range(i+1, len_):
            if a[i] == a[j] and b[i] == b[j]:
                countTrue += 1
            else:
                if (a[i] - a[j]) * (b[i] - b[j]) > 0:
                    countTrue += 1
                else:
                    countFalse += 1

    return countTrue, countFalse


@nb.njit
def _get_tf_score(WEIGHT, INDEX, PROFIT):
    '''
    Output: TrFScr
    '''
    countTrue = 0
    countFalse = 0
    for i in range(1, INDEX.shape[0] - 1):
        start, end = INDEX[i], INDEX[i+1]
        t, f = countTrueFalse(WEIGHT[start:end], PROFIT[start:end])
        countTrue += t
        countFalse += f

    return countTrue / (countFalse + 1e-6)


@nb.njit
def get_tf_score(WEIGHT, INDEX, PROFIT, SYMBOL, interest, num_cycle):
    '''
    Output: TrFScr
    '''
    result = []
    for i in range(num_cycle):
        ii = num_cycle - 1 - i
        WEIGHT_ = WEIGHT[INDEX[ii]:]
        PROFIT_ = PROFIT[INDEX[ii]:]
        INDEX_ = INDEX[ii:] - INDEX[ii]
        TrFScr = _get_tf_score(WEIGHT_, INDEX_, PROFIT_)
        result.append(TrFScr)

    return result


@nb.njit
def calculate_ac_coef(arr):
    if len(arr) < 2: return 0.0
    sum_ = 0.0
    l = len(arr)
    for i in range(l - 1):
        a = arr[i]
        b = arr[i+1:]
        nume = a - b
        deno = np.abs(a) + np.abs(b)
        deno[deno == 0.0] = 1.0
        sum_ += np.sum(nume/deno)

    result = sum_ / (l*(l-1))
    return max(result, 0.0)


@nb.njit
def _get_ac_score(WEIGHT, INDEX, PROFIT):
    '''
    Output: AccScr
    '''
    size = INDEX.shape[0]-1
    arr_coef = np.zeros(size-1)

    for i in range(size-1, 0, -1):
        start, end = INDEX[i], INDEX[i+1]
        weight_ = WEIGHT[start:end]
        profit_ = PROFIT[start:end]
        mask = weight_ != -1.7976931348623157e+308
        weight = weight_[mask]
        profit = profit_[mask]
        weight = weight[np.argsort(profit)[::-1]]
        arr_coef[size-1-i] = calculate_ac_coef(weight)
        if arr_coef[size-1-i] == 0.0: break

    return geomean(arr_coef)


@nb.njit
def get_ac_score(WEIGHT, INDEX, PROFIT, SYMBOL, interest, num_cycle):
    '''
    Output: AccScr
    '''
    result = []
    for i in range(num_cycle):
        ii = num_cycle - 1 - i
        WEIGHT_ = WEIGHT[INDEX[ii]:]
        PROFIT_ = PROFIT[INDEX[ii]:]
        INDEX_ = INDEX[ii:] - INDEX[ii]
        AccScr = _get_ac_score(WEIGHT_, INDEX_, PROFIT_)
        result.append(AccScr)

    return result


@nb.njit
def _get_inv_ngn2_infor(WEIGHT, INDEX, PROFIT, SYMBOL, interest):
    '''
    Output: Nguong2, ProNgn2, GeoNgn2, HarNgn2
    '''
    size = INDEX.shape[0] - 1
    arr_profit = np.zeros(size-2)
    temp_profit = np.zeros(size-2)
    max_profit = -1.0
    last_reason = 0

    list_loop = np.zeros((size-1)*5)
    for k in range(size-1, 0, -1):
        start, end = INDEX[k], INDEX[k+1]
        temp_weight = WEIGHT[start:end].copy()
        temp_weight[::-1].sort()
        list_loop[5*(k-1):5*k] = temp_weight[:5]

    list_loop = np.unique(list_loop)
    for v in list_loop:
        temp_profit[:] = 0.0
        reason = 0
        isbg = WEIGHT > v
        for i in range(size - 2):
            start, end = INDEX[-i-3], INDEX[-i-2]
            inv_cyc_val = isbg[start:end]
            if reason == 0:
                inv_cyc_sym = SYMBOL[start:end]
                pre_cyc_val = isbg[end:INDEX[-i-1]]
                pre_cyc_sym = SYMBOL[end:INDEX[-i-1]]
                coms = np.intersect1d(pre_cyc_sym[pre_cyc_val], inv_cyc_sym[inv_cyc_val])
                isin = np.full(end-start, False)
                for ii in range(end-start):
                    if inv_cyc_sym[ii] in coms:
                        isin[ii] = True
                lst_pro = PROFIT[start:end][isin]
            else:
                lst_pro = PROFIT[start:end][inv_cyc_val]

            if len(lst_pro) == 0:
                temp_profit[i] = interest
                if np.count_nonzero(inv_cyc_val) == 0:
                    reason = 1
            else:
                temp_profit[i] = np.mean(lst_pro)
                reason = 0

        new_profit = geomean(temp_profit)
        if new_profit > max_profit:
            Nguong2 = v
            max_profit = new_profit
            arr_profit[:] = temp_profit
            last_reason = reason

    isbg = WEIGHT > Nguong2
    start, end = INDEX[0], INDEX[1]
    inv_cyc_val = isbg[start:end]
    if last_reason == 0:
        inv_cyc_sym = SYMBOL[start:end]
        pre_cyc_val = isbg[end:INDEX[2]]
        pre_cyc_sym = SYMBOL[end:INDEX[2]]
        coms = np.intersect1d(pre_cyc_sym[pre_cyc_val], inv_cyc_sym[inv_cyc_val])
        isin = np.full(end-start, False)
        for ii in range(end-start):
            if inv_cyc_sym[ii] in coms:
                isin[ii] = True
        lst_pro = PROFIT[start:end][isin]
    else:
        lst_pro = PROFIT[start:end][inv_cyc_val]

    if len(lst_pro) == 0:
        ProNgn2 = interest
    else:
        ProNgn2 = np.mean(lst_pro)

    return Nguong2, ProNgn2, max_profit, harmean(arr_profit)


@nb.njit
def get_inv_ngn2_infor(WEIGHT, INDEX, PROFIT, SYMBOL, interest, num_cycle):
    '''
    Output: Nguong2, ProNgn2, GeoNgn2, HarNgn2
    '''
    result = []
    for i in range(num_cycle):
        ii = num_cycle - 1 - i
        WEIGHT_ = WEIGHT[INDEX[ii]:]
        PROFIT_ = PROFIT[INDEX[ii]:]
        INDEX_ = INDEX[ii:] - INDEX[ii]
        SYMBOL_ = SYMBOL[INDEX[ii]:]
        Nguong2, ProNgn2, GeoNgn2, HarNgn2 = _get_inv_ngn2_infor(WEIGHT_, INDEX_, PROFIT_, SYMBOL_, interest)
        result.append([Nguong2, ProNgn2, GeoNgn2, HarNgn2])

    return result


@nb.njit
def _get_inv_ngn1_2_infor(WEIGHT, INDEX, PROFIT, interest):
    '''
    Output: Nguong1_2, ProNgn1_2, GeoNgn1_2, HarNgn1_2
    '''
    size = INDEX.shape[0] - 1
    Nguong1_2 = -1.7976931348623157e+308
    for i in range(size-1, 0, -1):
        start, end = INDEX[i], INDEX[i+1]
        values = WEIGHT[start:end]
        arrPro = PROFIT[start:end]
        mask = np.argsort(arrPro)
        n = int(np.ceil(float(len(mask)) / 5))
        ngn = np.max(values[mask[:n]])
        if ngn > Nguong1_2:
            Nguong1_2 = ngn

    C = WEIGHT > Nguong1_2
    temp_profit = np.zeros(size-1)
    for i in range(size-1, 0, -1):
        start, end = INDEX[i], INDEX[i+1]
        if np.count_nonzero(C[start:end]) == 0:
            temp_profit[i-1] = interest
        else:
            temp_profit[i-1] = PROFIT[start:end][C[start:end]].mean()

    GeoNgn1_2 = geomean(temp_profit)
    HarNgn1_2 = harmean(temp_profit)

    start, end = INDEX[0], INDEX[1]
    mask = WEIGHT[start:end] > Nguong1_2
    if np.count_nonzero(mask) == 0:
        ProNgn1_2 = interest
    else:
        ProNgn1_2 = PROFIT[start:end][mask].mean()

    return Nguong1_2, ProNgn1_2, GeoNgn1_2, HarNgn1_2


@nb.njit
def get_inv_ngn1_2_infor(WEIGHT, INDEX, PROFIT, SYMBOL, interest, num_cycle):
    '''
    Output: Nguong1_2, ProNgn1_2, GeoNgn1_2, HarNgn1_2
    '''
    result = []
    for i in range(num_cycle):
        ii = num_cycle - 1 - i
        WEIGHT_ = WEIGHT[INDEX[ii]:]
        PROFIT_ = PROFIT[INDEX[ii]:]
        INDEX_ = INDEX[ii:] - INDEX[ii]
        Nguong1_2, ProNgn1_2, GeoNgn1_2, HarNgn1_2 = _get_inv_ngn1_2_infor(WEIGHT_, INDEX_, PROFIT_, interest)
        result.append([Nguong1_2, ProNgn1_2, GeoNgn1_2, HarNgn1_2])

    return result


@nb.njit
def _get_inv_ngn1_3_infor(WEIGHT, INDEX, PROFIT, interest):
    '''
    Output: Nguong1_3, ProNgn1_3, GeoNgn1_3, HarNgn1_3
    '''
    size = INDEX.shape[0] - 1
    start = INDEX[1]
    mask = PROFIT[start:] < 1.0
    Nguong1_3 = np.max(WEIGHT[start:][mask])

    C = WEIGHT > Nguong1_3
    temp_profit = np.zeros(size-1)
    for i in range(size-1, 0, -1):
        start, end = INDEX[i], INDEX[i+1]
        if np.count_nonzero(C[start:end]) == 0:
            temp_profit[i-1] = interest
        else:
            temp_profit[i-1] = PROFIT[start:end][C[start:end]].mean()

    GeoNgn1_3 = geomean(temp_profit)
    HarNgn1_3 = harmean(temp_profit)

    start, end = INDEX[0], INDEX[1]
    mask = WEIGHT[start:end] > Nguong1_3
    if np.count_nonzero(C[start:end]) == 0:
        ProNgn1_3 = interest
    else:
        ProNgn1_3 = PROFIT[start:end][mask].mean()

    return Nguong1_3, ProNgn1_3, GeoNgn1_3, HarNgn1_3


@nb.njit
def get_inv_ngn1_3_infor(WEIGHT, INDEX, PROFIT, SYMBOL, interest, num_cycle):
    '''
    Output: Nguong1_3, ProNgn1_3, GeoNgn1_3, HarNgn1_3
    '''
    result = []
    for i in range(num_cycle):
        ii = num_cycle - 1 - i
        WEIGHT_ = WEIGHT[INDEX[ii]:]
        PROFIT_ = PROFIT[INDEX[ii]:]
        INDEX_ = INDEX[ii:] - INDEX[ii]
        Nguong1_3, ProNgn1_3, GeoNgn1_3, HarNgn1_3 = _get_inv_ngn1_3_infor(WEIGHT_, INDEX_, PROFIT_, interest)
        result.append([Nguong1_3, ProNgn1_3, GeoNgn1_3, HarNgn1_3])

    return result
