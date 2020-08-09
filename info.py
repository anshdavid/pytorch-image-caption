# -*- coding: utf-8 -*-

from torchvision import models
try:
    from torchsummary import summary
except:
    MODELSUMMARY = False
else:
    MODELSUMMARY = True


if __name__ == "__main__":

    input_shape = tuple
    model_ = None

    print(f"-----------model-----------\n{model_}\n---------------------------\n\n")

    if MODELSUMMARY and input_shape:
        summary(model_, input_data=input_shape)     #type:ignore

