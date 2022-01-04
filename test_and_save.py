from PIL import Image
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def test_paint(pic_name, device, NET):
    with torch.no_grad():
        pic_path = pic_name 
        with Image.open(pic_path) as im:
            pic_transform = transforms.ToTensor()
            pic = pic_transform(im)
            im = np.array(im)
            plt.imshow(im, cmap='gray')
            plt.axis('off')
            plt.show()

            gray_test = pic
            if pic.shape[0] > 1:
                gray_transform = torchvision.transforms.Grayscale(
                    num_output_channels=1)
                gray_test = gray_transform(pic)
            gray_test = torch.unsqueeze(gray_test, 0).to(device)

            outputs = NET(gray_test)

            im = save(outputs[0], pic_name)
            plt.imshow(im)
            plt.axis('off')

            del outputs
            del gray_test
            torch.cuda.empty_cache()





def save(img, pic_name):
    npimg = img.cpu().numpy()
    npimg = np.clip(npimg, 0, 1)
    npimg = np.transpose(npimg, (1, 2, 0))
    im = Image.fromarray(((npimg * 255).astype(np.uint8)))
    im.save("{}_pred.jpg".format(pic_name))
    return im
