import os
from glob import glob
import time
rootpath=r'.\data\faces'
start_time = time.time() # 计时开始
f=glob(os.path.join(rootpath,'*.jpg'))
for i in range(len(f)):
    os.rename(f[i],os.path.join(rootpath,str(i+1)+'.jpg'))
    end_time = time.time()  # 结束时间
    print(end_time - start_time)  # 打印消耗的时间

