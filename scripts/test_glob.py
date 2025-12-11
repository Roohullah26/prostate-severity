import glob, pprint
pprint.pprint(glob.glob('data/PROSTATEx/**/ProstateX-0222*', recursive=True))
