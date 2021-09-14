import _PyPacwar
import numpy as np
import time
import heapq
##function to get value
def valfun(gene):
    #[0]*50,,[2]*50
    opponent=[[1]*50,[3]*50,
              [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 0, 1, 1, 1],
              [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 3, 1],
              [1, 3, 2, 2, 3, 0, 0, 1, 1, 2, 3, 3, 0, 2, 3, 0, 3, 2, 0, 3, 1, 3, 1, 2, 2, 3, 3, 1, 1, 1, 1, 1, 0, 0, 3, 1, 3, 1, 1, 1, 0, 0, 2, 1, 3, 2, 0, 1, 0, 2],
              [1, 0, 1, 1, 0, 1, 1, 3, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 3, 1, 0, 1, 1, 1, 1, 1, 1]]
    val = 0
    i = 0
    for player in opponent:
        (rounds, c1, c2) = _PyPacwar.battle(gene, player)
        val+=score(rounds,c1,c2)
        i+=1
    return val/i



def score(rounds, c1, c2):
    if c2 == 0 and c1 > 0:
        if rounds < 100:
            value = 20
        elif rounds < 200:
            value = 18
        elif rounds < 300:
            value = 16
        elif rounds <= 500:
            value = 14
    elif c1 / c2 >= 10:
        value = 6
    elif c1 / c2 >= 3:
        value = 4
    elif c1 / c2 >= 1.5:
        value = 2
    else:
        value = 0
    return value


start = time.time()
init = [1]*50
num = 5 #number of restart
result = np.zeros((num,51))
L = 1000
for i in range(0,num):
    print('starting iteration %d '%(i))
    curr = init
    # if i == 0:
    #     curr = init
    # else:
    #     curr = np.random.randint(4,size=50)
    #     curr = list(curr)
    val = 0
    for j in range(L):
        if j%100==0:
            print('change %d'%j)
        Good=[]
        Notgood=[]
        if j==0:
            val_new=valfun(curr)
            if val_new == 20.0:
                result[i][0]=val_new
                result[i][1:]=curr
                break
        else:
            for k in range(50):
                for plus in range(1,4):
                    temp = curr[:]
                    temp[k]=(temp[k]+plus)%4
                    val_new = valfun(temp)
                    if val_new == 20:
                        result[i][0] = val_new
                        result[i][1:] = curr
                        break
                    if val_new>val:
                        temp.append(val_new)
                        Good.append(temp)
                    elif Notgood:
                        if -Notgood[0][0]<val_new:
                            heapq.heappush(Notgood,(-val_new,tuple(temp)))
                    else:
                        heapq.heappush(Notgood, (-val_new, tuple(temp)))
                if val_new== 20.0:
                    break
            if not Good and val_new!=20.0:
                key,gene=heapq.heappop(Notgood)
                gene = list(gene)
                if Notgood:
                    key2,gene2=heapq.heappop(Notgood)
                    while key == key2:
                        gene2 = list(gene)
                        gene2.append(-key2)
                        Good.append(gene2)
                        if Notgood:
                            key2, gene2 = heapq.heappop(Notgood)
                gene.append(-key)
                Good.append(gene)
            if val_new == 20.0:
                break
            n = len(Good)
            rand=np.random.randint(n, size=1)

            curr=Good[int(rand)][:50]
            val_new=Good[int(rand)][50]
        val=val_new
    result[i][0]=val
    result[i][1:]=curr
    print(curr)
    print(val)
np.savetxt("candidate911_2.csv", result, delimiter=",")
end=time.time()
print(end-start)