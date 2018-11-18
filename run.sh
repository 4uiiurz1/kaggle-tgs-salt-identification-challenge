python train.py --name SCSECatResUNet34 --arch SCSECatResUNet34
python validate.py --name SCSECatResUNet34
python test.py --name SCSECatResUNet34

python pseudo_train.py --name SCSECatResUNet34_pseudo --org-name SCSECatResUNet34
python validate.py --name SCSECatResUNet34_pseudo
python test.py --name SCSECatResUNet34_pseudo

python train.py --name SCSECatSEResNeXt32x4dUNet50 --arch SCSECatSEResNeXt32x4dUNet50
python validate.py --name SCSECatSEResNeXt32x4dUNet50
python test.py --name SCSECatSEResNeXt32x4dUNet50

python pseudo_train.py --name SCSECatSEResNeXt32x4dUNet50_pseudo --org-name SCSECatSEResNeXt32x4dUNet50
python validate.py --name SCSECatSEResNeXt32x4dUNet50_pseudo
python test.py --name SCSECatSEResNeXt32x4dUNet50_pseudo

python train.py --name SCSECatResUNet34_pad --arch SCSECatResUNet34 --pad True
python validate.py --name SCSECatResUNet34_pad
python test.py --name SCSECatResUNet34_pad

python pseudo_train.py --name SCSECatResUNet34_pad_pseudo --org-name SCSECatResUNet34_pad
python validate.py --name SCSECatResUNet34_pad_pseudo
python test.py --name SCSECatResUNet34_pad_pseudo

python averaging.py
