import numpy as np

available_weights = ['ffhq', 'car', 'cat', 'church', 'horse']
weights_stylegan2_dir = 'weights/'

mapping_weights = [ 'Dense0/weight', 'Dense0/bias',
                    'Dense1/weight', 'Dense1/bias',
                    'Dense2/weight', 'Dense2/bias',
                    'Dense3/weight', 'Dense3/bias',
                    'Dense4/weight', 'Dense4/bias',
                    'Dense5/weight', 'Dense5/bias',
                    'Dense6/weight', 'Dense6/bias',
                    'Dense7/weight', 'Dense7/bias']

def get_synthesis_name_weights(resolution):
    synthesis_weights = ['4x4/Const/const',
                         '4x4/Conv/noise_strength',
                         '4x4/Conv/bias',
                         '4x4/Conv/mod_bias',
                         '4x4/Conv/mod_weight',
                         '4x4/Conv/weight',
                         '4x4/ToRGB/bias',
                         '4x4/ToRGB/mod_bias',
                         '4x4/ToRGB/mod_weight',
                         '4x4/ToRGB/weight']

    for res in range(3,int(np.log2(resolution)) + 1):
        name = '{}x{}/'.format(2**res, 2**res)
        for up in ['Conv0_up/', 'Conv1/', 'ToRGB/']:
            for var in ['noise_strength', 'bias', 'mod_bias', 'mod_weight', 'weight']:
                if up == 'ToRGB/' and var == 'noise_strength':
                    continue
                synthesis_weights.append(name+up+var)
                
    return synthesis_weights

synthesis_weights_1024 = get_synthesis_name_weights(1024)
synthesis_weights_512 = get_synthesis_name_weights(512)
synthesis_weights_256 = get_synthesis_name_weights(256)


discriminator_weights_1024 = ['disc_4x4/Conv/bias',
                            'disc_1024x1024/FromRGB/bias',
                            'disc_1024x1024/FromRGB/weight',
                            'disc_1024x1024/Conv0/bias',
                            'disc_1024x1024/Conv1_down/bias',
                            'disc_1024x1024/Conv0/weight',
                            'disc_1024x1024/Conv1_down/weight',
                            'disc_1024x1024/Skip/weight',
                            'disc_512x512/Conv0/bias',
                            'disc_512x512/Conv1_down/bias',
                            'disc_512x512/Conv0/weight',
                            'disc_512x512/Conv1_down/weight',
                            'disc_512x512/Skip/weight',
                            'disc_256x256/Conv0/bias',
                            'disc_256x256/Conv1_down/bias',
                            'disc_256x256/Conv0/weight',
                            'disc_256x256/Conv1_down/weight',
                            'disc_256x256/Skip/weight',
                            'disc_128x128/Conv0/bias',
                            'disc_128x128/Conv1_down/bias',
                            'disc_128x128/Conv0/weight',
                            'disc_128x128/Conv1_down/weight',
                            'disc_128x128/Skip/weight',
                            'disc_64x64/Conv0/bias',
                            'disc_64x64/Conv1_down/bias',
                            'disc_64x64/Conv0/weight',
                            'disc_64x64/Conv1_down/weight',
                            'disc_64x64/Skip/weight',
                            'disc_32x32/Conv0/bias',
                            'disc_32x32/Conv1_down/bias',
                            'disc_32x32/Conv0/weight',
                            'disc_32x32/Conv1_down/weight',
                            'disc_32x32/Skip/weight',
                            'disc_16x16/Conv0/bias',
                            'disc_16x16/Conv1_down/bias',
                            'disc_16x16/Conv0/weight',
                            'disc_16x16/Conv1_down/weight',
                            'disc_16x16/Skip/weight',
                            'disc_8x8/Conv0/bias',
                            'disc_8x8/Conv1_down/bias',
                            'disc_8x8/Conv0/weight',
                            'disc_8x8/Conv1_down/weight',
                            'disc_8x8/Skip/weight',
                            'disc_4x4/Conv/weight',
                            'disc_4x4/Dense0/weight',
                            'disc_4x4/Dense0/bias',
                            'disc_Output/weight',
                            'disc_Output/bias']

discriminator_weights_512 = ['disc_4x4/Conv/bias',
                            'disc_512x512/FromRGB/bias',
                            'disc_512x512/FromRGB/weight',
                            'disc_512x512/Conv0/bias',
                            'disc_512x512/Conv1_down/bias',
                            'disc_512x512/Conv0/weight',
                            'disc_512x512/Conv1_down/weight',
                            'disc_512x512/Skip/weight',
                            'disc_256x256/Conv0/bias',
                            'disc_256x256/Conv1_down/bias',
                            'disc_256x256/Conv0/weight',
                            'disc_256x256/Conv1_down/weight',
                            'disc_256x256/Skip/weight',
                            'disc_128x128/Conv0/bias',
                            'disc_128x128/Conv1_down/bias',
                            'disc_128x128/Conv0/weight',
                            'disc_128x128/Conv1_down/weight',
                            'disc_128x128/Skip/weight',
                            'disc_64x64/Conv0/bias',
                            'disc_64x64/Conv1_down/bias',
                            'disc_64x64/Conv0/weight',
                            'disc_64x64/Conv1_down/weight',
                            'disc_64x64/Skip/weight',
                            'disc_32x32/Conv0/bias',
                            'disc_32x32/Conv1_down/bias',
                            'disc_32x32/Conv0/weight',
                            'disc_32x32/Conv1_down/weight',
                            'disc_32x32/Skip/weight',
                            'disc_16x16/Conv0/bias',
                            'disc_16x16/Conv1_down/bias',
                            'disc_16x16/Conv0/weight',
                            'disc_16x16/Conv1_down/weight',
                            'disc_16x16/Skip/weight',
                            'disc_8x8/Conv0/bias',
                            'disc_8x8/Conv1_down/bias',
                            'disc_8x8/Conv0/weight',
                            'disc_8x8/Conv1_down/weight',
                            'disc_8x8/Skip/weight',
                            'disc_4x4/Conv/weight',
                            'disc_4x4/Dense0/weight',
                            'disc_4x4/Dense0/bias',
                            'disc_Output/weight',
                            'disc_Output/bias']

discriminator_weights_256 =  ['disc_4x4/Conv/bias',
                            'disc_256x256/FromRGB/bias',
                            'disc_256x256/FromRGB/weight',
                            'disc_256x256/Conv0/bias',
                            'disc_256x256/Conv1_down/bias',
                            'disc_256x256/Conv0/weight',
                            'disc_256x256/Conv1_down/weight',
                            'disc_256x256/Skip/weight',
                            'disc_128x128/Conv0/bias',
                            'disc_128x128/Conv1_down/bias',
                            'disc_128x128/Conv0/weight',
                            'disc_128x128/Conv1_down/weight',
                            'disc_128x128/Skip/weight',
                            'disc_64x64/Conv0/bias',
                            'disc_64x64/Conv1_down/bias',
                            'disc_64x64/Conv0/weight',
                            'disc_64x64/Conv1_down/weight',
                            'disc_64x64/Skip/weight',
                            'disc_32x32/Conv0/bias',
                            'disc_32x32/Conv1_down/bias',
                            'disc_32x32/Conv0/weight',
                            'disc_32x32/Conv1_down/weight',
                            'disc_32x32/Skip/weight',
                            'disc_16x16/Conv0/bias',
                            'disc_16x16/Conv1_down/bias',
                            'disc_16x16/Conv0/weight',
                            'disc_16x16/Conv1_down/weight',
                            'disc_16x16/Skip/weight',
                            'disc_8x8/Conv0/bias',
                            'disc_8x8/Conv1_down/bias',
                            'disc_8x8/Conv0/weight',
                            'disc_8x8/Conv1_down/weight',
                            'disc_8x8/Skip/weight',
                            'disc_4x4/Conv/weight',
                            'disc_4x4/Dense0/weight',
                            'disc_4x4/Dense0/bias',
                            'disc_Output/weight',
                            'disc_Output/bias']

synthesis_weights = {
    'ffhq' : synthesis_weights_1024,
    'car' : synthesis_weights_512,
    'cat' : synthesis_weights_256,
    'horse' : synthesis_weights_256,
    'church' : synthesis_weights_256
    }

discriminator_weights = {
    'ffhq' : discriminator_weights_1024,
    'car' : discriminator_weights_512,
    'cat' : discriminator_weights_256,
    'horse' : discriminator_weights_256,
    'church' : discriminator_weights_256
    }