# Topology W1 assignment
# Huajun SUN
import numpy as np
# x = [[0,0],
#      [1,1]]
# size = [2,2]
x = [[0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 1, 1, 1, 0],
     [0, 0, 1, 1, 0, 1, 0],
     [0, 1, 1, 1, 0, 1, 0],
     [0, 1, 0, 1, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0]]
size = [6, 7]


def inverse_x(b):
    """
    :param b: a binary matrix
    :return: r_x: the inverse for x
    """
    return b ^ np.ones_like(b)


def find_connection_4(t):
    """
    :param t: 3*3 binary matrix
    :return: n: the number of connectivity number of 4_connect
    """
    mark = np.zeros_like(t)
    label = 0
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue  # skip the central point
            if t[i][j] == 0:
                continue
            else:  # it meets one connect area
                # get the labels of the left and up
                if j - 1 >= 0:
                    r_label = mark[i][j - 1]
                else:
                    r_label = 0
                if i - 1 >= 0:
                    u_label = mark[i - 1][j]
                else:
                    u_label = 0

                if r_label or u_label:
                    if r_label > 0 and u_label > 0:
                        mark[i][j] = min(r_label, u_label)  # get the smallest label
                    else:
                        mark[i][j] = max(r_label, u_label)  # get the label than is bigger than 0
                else:
                    label += 1  # label has to increase
                    mark[i][j] = label

    # backwards to see if there is 2-layer trees.
    for i in range(2, -1, -1):
        for j in range(2, -1, -1):
            if i == 1 and j == 1:
                continue  # skip the central point
            if mark[i][j]:
                # it meets one connect area
                # get the labels of the left and up
                if j + 1 < 3:
                    r_label = mark[i][j + 1]
                else:
                    r_label = 0
                if i + 1 < 3:
                    u_label = mark[i + 1][j]
                else:
                    u_label = 0

                if r_label or u_label:
                    if r_label > 0 and u_label > 0:
                        mark[i][j] = min(r_label, u_label)  # get the smallest label
                    else:
                        mark[i][j] = max(r_label, u_label)  # get the label than is bigger than 0
    
    #print("mark is :\n{}".format(mark))
    l = [mark[0][1], mark[1][0], mark[1][2], mark[2][1]]
    s = set(l)
    s.discard(0)
    # n = np.max(mark)
    return len(s)


def find_connection_8(t,flag):
    """
    :param t: 3*3 binary matrix
    :return: n: the number of connectivity number of 4_connect
    """
    mark = np.zeros_like(t)
    label = 0
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                continue  # skip the central point
            if t[i][j] == 0:
                continue
            else:  # it meets one connect area
                # get the labels of the left and up, and 1030 o'clock
                if j - 1 >= 0:
                    l_label = mark[i][j - 1]
                else:
                    l_label = 0
                if i - 1 >= 0:
                    u_label = mark[i - 1][j]
                else:
                    u_label = 0
                if j - 1 >= 0 and i - 1 >= 0:
                    lu_label = mark[i - 1][j - 1]
                else:
                    lu_label = 0
                if j + 1 <3  and i - 1 >= 0:
                    ru_label = mark[i - 1][j + 1]
                else:
                    ru_label = 0

                if l_label or u_label or lu_label or ru_label:
                    if min(l_label, u_label, lu_label,ru_label):
                        mark[i][j] = min(l_label, u_label, lu_label,ru_label)
                    else:
                        m = 10000
                        for z in [l_label, u_label, lu_label,ru_label]:
                            if 0 < z < m: m = z
                        mark[i][j] = m
                    # get the smallest label that is bigger than 0
                else:
                    label += 1  # label has to increase
                    mark[i][j] = label

    # backwards to see if there is 2-layer trees.
    for i in range(2, -1, -1):
        for j in range(2, -1, -1):
            if i == 1 and j == 1:
                continue  # skip the central point
            if mark[i][j]:
                # it meets one connect area
                # get the labels of the right and up and right-up: 430
                if j + 1 < 3:
                    r_label = mark[i][j + 1]
                else:
                    r_label = 0
                if i + 1 <3:
                    u_label = mark[i + 1][j]
                else:
                    u_label = 0
                if j + 1 < 3 and i + 1 < 3:
                    rd_label = mark[i + 1][j + 1]
                else:
                    rd_label = 0
                if j - 1 >=0 and i + 1 < 3:
                    ld_label = mark[i + 1][j - 1]
                else:
                    ld_label = 0

                if r_label or u_label or rd_label or ld_label:
                    if min(r_label, u_label, rd_label,ld_label):
                        mark[i][j] = min(r_label, u_label, rd_label,ld_label)
                    else:
                        m = 10000
                        for z in [r_label, u_label, rd_label,ld_label]:
                            if 0 < z < m:
                                m = z
                        mark[i][j] = m
                    # get the smallest label that is bigger than 0
                    if r_label > 0 and u_label > 0:
                        mark[i][j] = min(r_label, u_label)  # get the smallest label
                    else:
                        mark[i][j] = max(r_label, u_label)  # get the label than is bigger than 0
    if flag:
        print("mark is :\n{}".format(mark))
    # l = [mark[0][1], mark[1][0], mark[1][2], mark[2][1]]
    # s = set(l)
    # s.discard(0)
    n = np.max(mark)
    return n


if __name__ == "__main__":

    count_4 = 0
    count_8 = 0
    res_4 = np.zeros(size)
    res_8 = np.zeros(size)
    inx = inverse_x(x)

    for i in range(size[0]):
        for j in range(size[1]):
            temp = np.zeros([3, 3])
            intemp = np.zeros([3, 3])
            for k in range(-1, 2):
                for b in range(-1, 2):
                    if i + k < 0 or i + k >= size[0]:
                        continue
                    if j + b < 0 or j + b >= size[1]:
                        continue
                    temp[k + 1][b + 1] = x[i + k][j + b]
                    intemp[k + 1][b + 1] = inx[i + k][j + b]
            if i==3 and (j==1 or j==4):
                print("temp is \n{}".format(temp))
            flag=True if i==3 and (j==1 or j==4) else False
            incon_4 = find_connection_4(intemp)
            con_4 = find_connection_4(temp)
            incon_8 = find_connection_8(intemp,flag)
            con_8 = find_connection_8(temp,flag)
            if con_4 == 1 and incon_4 == 1:
                res_4[i][j] = 1
                # print(1)
                count_4 += 1
            if con_8 == 1 and incon_8 == 1:
                res_8[i][j] = 1
                # print(1)
                count_8 += 1
            # else: print(0)

        # print('\n')
    # print(np.array(x))
    # print(inx)
    # # print(count_4)
    print(count_8)
    # # print(res_4)
    print(res_8)
