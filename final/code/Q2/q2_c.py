x=0
while True:
    belta = 0.9
    u = [100] * 10
    flag = 1
    cnt = 0
    while flag:
        cnt += 1
        l = []
        for i in range(10):
            if i == 0:
                l.append(101 + belta * u[1])
            elif i == 9:
                l.append(max(-255 + belta * u[0], -x+belta*(0.5*u[1]+0.5*u[2])))
            else:
                l.append(max(100 - 10 * i + belta * (0.1 * i * u[i + 1] + (1 - 0.1 * i) * u[i]), -255 + belta * u[0],-x+belta*(0.5*u[1]+0.5*u[2])))
        flag = 0
        for i, j in zip(l, u):
            if abs(i - j) > 0.001:
                flag = 1
                break
        u = l.copy()
    i=8
    if -255 + belta * u[0] > -x+belta*(0.5*u[1]+0.5*u[2]):
        break
    else:
        x+=1
print('the highest price: ', x-1)
        # print('cnt: ', cnt, 'utility: ', u)



