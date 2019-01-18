require 'loadcaffe'
require 'xlua'
require 'optim'

—- modify the path 

prototxt = 'VGG_ILSVRC_19_layers_deploy.prototxt'
binary = 'VGG_ILSVRC_19_layers.caffemodel'

net = loadcaffe.load(prototxt, binary, 'cudnn')
net = net:float() —- essential reference https://github.com/clcarwin/convert_torch_to_pytorch/issues/8
print(net)

torch.save('vgg16_torch.t7', net)
