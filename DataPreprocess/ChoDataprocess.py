import json
import pandas as pd

starArr={}
#counter = 0
with open('yelp_academic_dataset_review.json', 'r', encoding='UTF8') as file:
    for line in file:
        #if(counter==200):
        #    break
        data = json.loads(line)

        #type(data)
        #print(data)
        #counter+=1
        if  data['business_id'] in starArr: #기존에 있다면 그냥 추가한다.
            starArr[data['business_id']].append(data['stars'])
        else: #없으면 키와 빈 배열을 추가.
            starArr[data['business_id']]=[]
            starArr[data['business_id']].append(data['stars'])

#print("count : ")
#print(counter)
#print(starArr)

bsArr=[]#회사코드
avrArr=[]
for b in starArr:
    bsArr.append(b)
    avrArr.append(sum(starArr[b])/len(starArr[b]))

#counter=0
bsnArr=['0']*len(bsArr)
with open('yelp_academic_dataset_business.json', 'r', encoding='UTF8') as file:
    for line in file:
        #if(counter==1):
        #    break
        data2 = json.loads(line)

        #type(data2)
        #print(data2)
        #counter+=1
        
        if data2['business_id'] in bsArr:
            bsnArr[bsArr.index(data2['business_id'])]=data2['name']


#if len(bsArr) != len(bsnArr):
#    print("COUNT ERR")
#    exit()



#=======================

df = pd.DataFrame(bsArr, columns = ['회사코드'])


df['평균'] = avrArr
df['업소명'] = bsnArr

df.to_csv("BS_STARS.csv", index = False, encoding="UTF-8-sig")
print("done")