'''from xenonpy.datatools import preset
import pandas as pd
preset.sync('elements_completed')
data=preset.elements_completed
newdata=pd.DataFrame(data,columns=['atomic_number']).T
newdata.to_json('atom.json')
data=pd.read_json('atom.json')
data'''
'''import torch
x=torch.randn(2,3,1)
y=torch.randn(2,3,4)
a=torch.cat((x,y),dim=2)
print(a.size())'''
import numpy as np
x=np.array([1,2,3])
print(x*10)