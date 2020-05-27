import os
os.system('cmd /c "pip install gdown"')

import gdown
    
file_id = ['1afMN3e_6UuTTPDL63WHaA0Fb9EQrZceE', '1Av4p3JNWkmWsJx6s9Hq32JvlNWSW4kgk',
	   '1LCyHcKtqNs8NirJeSPq3fKAUZ-MLt_-8', '1pEAeJTK_ZPfkUvV7VFDHq0O4tKYsslkX', 
	   '16_QtKS2w9-40uxGZ02h3OtFhnJgNgdi_']
name = ['ffhq.npy', 'car.npy', 'cat.npy', 'church.npy', 'horse.npy']

weight_dir = './'
for i in range(len(name)):
    output = weight_dir + name[i]
    url = "https://drive.google.com/uc?id={}".format(file_id[i])
    gdown.download(url, output, quiet=False) 