from src.prostate_dataset import ProstateLesionDataset
from src.utils_image import get_train_tensor_transform

def run():
    print('SMOKE TEST - dataset imports')
    ds = ProstateLesionDataset(toy=True, toy_len=4, num_slices=3)
    print('toy single seq - num_channels =', ds.num_channels)
    img, label = ds[0]
    print('sample type:', type(img), 'shape:', getattr(img, 'shape', None), 'label', label)

    ds2 = ProstateLesionDataset(toy=True, toy_len=4, sequences=['t2','adc'], num_slices=3)
    print('toy multi-seq - num_channels =', ds2.num_channels)
    img2, label2 = ds2[0]
    print('sample type:', type(img2), 'shape:', getattr(img2, 'shape', None), 'label', label2)

    print('apply augmentation transform')
    trans = get_train_tensor_transform()
    out = trans(img2)
    print('after transform shape:', getattr(out, 'shape', None))

if __name__ == '__main__':
    run()
