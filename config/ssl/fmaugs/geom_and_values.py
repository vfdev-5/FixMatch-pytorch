import fmaugs

fmaugs_type = "conv+avgpool;"

ops = fmaugs.Compose([
    fmaugs.RandomOpMaskSamplewise(fmaugs.Noise(factor=4.5), density=0.5, p=0.5),
    fmaugs.RandomOpMaskSamplewise(fmaugs.Flip("h"), density=0.3, p=0.5),
    fmaugs.RandomOpMaskSamplewise(fmaugs.Roll(0.3), density=0.5, p=0.5),
])
