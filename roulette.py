import numpy as np

def roulette(probability):
    probabilityTotal = np.zeros(len(probability))
    probabilityTmp = 0
    for i in range(len(probability)):
        probabilityTmp += probability[i]
        probabilityTotal[i] = probabilityTmp
    randomNumber=np.random.rand()
    result=0
    for i in range(1, len(probabilityTotal)):
        if randomNumber<probabilityTotal[0]:
            result=0
            print("random number:",randomNumber,"<index 0:",probabilityTotal[0])
            break
        elif probabilityTotal[i - 1] < randomNumber <= probabilityTotal[i]:
            result=i
            print("index ",i-1,":",probabilityTotal[i-1],"<random number:",randomNumber,"<index ",i,":",probabilityTotal[i])
    return result

probability=[0.39897898,0.05301439,0.03615747,0.02251049,0.02993633,0.02285509,0.01361222,0.03879427,0.01937278,0.0274668,0.02173864,0.03050861,0.01137698,0.00821177,0.01364433,0.01546327,0.05063628,0.03238738,0.03441008,0.01896785,0.01343442,0.01250853,0.00500341,0.01080083,0.01227628,0.00865885,0.01000899,0.01326119,0.0140035]
result=np.zeros(100).astype(int)
count=np.zeros(100).astype(int)
for i in range(100):
    print("loop",i)
    result[i]=roulette(probability)
    count[result[i]]+=1
print("result:",result)
print("count:",count)
