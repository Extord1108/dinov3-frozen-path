from dinov3.data.datasets import Pathology

for split in Pathology.Split:
    dataset = Pathology(split=split, root="./dataset/pathology/", extra="./dataset/pathology/")
    dataset.dump_extra()