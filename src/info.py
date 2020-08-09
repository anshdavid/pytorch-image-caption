# -*- coding: utf-8 -*-

try:
    from torchsummary import summary
except:
    MODELSUMMARY = False
else:
    MODELSUMMARY = True


def modelInfo(model, input_shape: tuple = None):

    print("-----------summary-----------\n")

    if MODELSUMMARY and input_shape:
        summary(model, input_data=input_shape)     #type:ignore

    print(f"-----------model-----------\n{model}\n---------------------------\n\n")


