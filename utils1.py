import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image
from skimage import io, transform
import scipy.misc
import torch
import itertools

imsize = 256

loader = transforms.Compose([
             transforms.Resize([imsize, imsize]),
             transforms.ToTensor()
         ])

unloader = transforms.ToPILImage()

def image_loader(image_name):
    image = Image.open(image_name)
    image_tensor = loader(image)
    image_tensor.unsqueeze_(0)
    image_variable = Variable(image_tensor)
    return image_variable
  
def save_image(image, path):
    image = image.view(3, imsize, imsize)
    image = unloader(image)
    scipy.misc.imsave(path, image)

def save_images(input, paths):
    N = input.size()[0]
    images = input.data.clone().cpu()
    for i in range(N):
        save_image(images[i], paths[i])

def get_content_and_style(loader1, loader2, num_iters):
    iter1 = itertools.cycle(loader1)
    iter2 = itertools.cycle(loader2)

    for _ in range(num_iters):
        yield (next(iter1), next(iter2))
        