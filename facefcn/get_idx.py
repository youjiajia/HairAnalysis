import os

f = open('../../datasets/INDO/indices.txt','w')
for file in os.listdir('../../datasets/INDO/img/'):
    names = []
    for i in range(1,5):
        f.write('indo ' + file + ' ' + str(i) + '\n')
    print file