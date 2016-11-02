import numpy as np

def num_to_nbrs(num):
    return np.array([(num & (1<<k)) >> k for k in range(8)])

def num_transitions(nbrs):
    return np.sum(np.diff(np.concatenate((nbrs, [nbrs[0]])))>0)

def lut_to_str(lut):
    return '{{{:s}}}'.format(','.join(str(i) for i in lut))

def nbrs_to_img(nbrs):
    reorder = np.concatenate((nbrs[[0, 1, 2, 6,]], [1], nbrs[[3, 6, 5, 4]]))
    return np.reshape(reorder, (3, 3)).astype('uint8')

def guo_hall_unit(num, iteration):
    nbrs = num_to_nbrs(num)
    NW, N, NE, E, SE, S, SW, W = nbrs

    C = num_transitions(nbrs)
    N1 = (NW | N) + (NE | E) + (SE | S) + (SW | W);
    N2 = (N | NE) + (E | SE) + (S | SW) + (W | NW);
    if iteration % 2 == 0:
        m = ((N | NE | (1 - SE)) | E)
    else:
        m = ((S | SW | (1 - NW)) & W)

    if C == 1 and (1 < min(N1, N2) < 4) and m == 0:
        return 0
    else:
        return 1

def zhang_suen_unit(num, iteration):
    nbrs = num_to_nbrs(num)
    NW, N, NE, E, SE, S, SW, W = nbrs
    A = num_transitions(nbrs)
    if iteration % 2 == 0:
        cond3 = N * E * S == 0
        cond4 = E * S * W == 0
    else:
        cond3 = W * N * E == 0
        cond4 = S * W * N == 0

    if (A == 1) and (2 <= np.sum(nbrs) <= 6) and cond3 and cond4:
        return 0
    else:
        return 1

def zhang_suen_unit_skimage(num, iteration):
    #lut from skimage - seems to be modified version of zhang zuen
    lut = [
        0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0,
        0, 0, 2, 0, 2, 0, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0,
        0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0, 0, 0, 3, 1,
        0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 3, 0, 0,
        1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3,
        0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0
    ]

    return 1-(lut[num] & iteration)

def lut_initializer_define(defname, lut):
    return '#define {} {}'.format(defname, lut_to_str(lut))

def main():
    zhangsuen_lut1 = [zhang_suen_unit(n, 1) for n in range(256)]
    zhangsuen_lut2 = [zhang_suen_unit(n, 2) for n in range(256)]

    guohall_lut1 = [guo_hall_unit(n, 1) for n in range(256)]
    guohall_lut2 = [guo_hall_unit(n, 2) for n in range(256)]

    branch_lut = [int(num_transitions(num_to_nbrs(n))>2) for n in range(256)]
    endpts_lut = [int(num_transitions(num_to_nbrs(n))==1) for n in range(256)]

    print(lut_initializer_define('ZHANGSUEN_LUT1', zhangsuen_lut1), end='\n\n')
    print(lut_initializer_define('ZHANGSUEN_LUT2', zhangsuen_lut2), end='\n\n')
    print(lut_initializer_define('GUOHALL_LUT1', guohall_lut1), end='\n\n')
    print(lut_initializer_define('GUOHALL_LUT2', guohall_lut2), end='\n\n')
    print(lut_initializer_define('BRANCH_LUT', branch_lut), end='\n\n')
    print(lut_initializer_define('ENDPTS_LUT', endpts_lut), end='\n\n')

if __name__ == '__main__':
    main()
