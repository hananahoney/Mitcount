# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf

# import cv2 as cv
# # Load model
# model = torch.load('latest_model.pt', map_location=torch.device('cpu'))

# image = Image.open('A06_00Aa.png')

# input_shape = (3, 384, 384)
# # transform = transforms.Compose([
# #     # transforms.Resize((256, 256)),
# #     transforms.ToTensor(),
# #     # transforms.Normalize(mean=[0.485, 0.456, 0.406],
# #                         #  std=[0.229, 0.224, 0.225])
# # ])

# transform = transforms.Compose([
#     transforms.Resize((input_shape[1], input_shape[2])),
#     transforms.ToTensor(),
# ])

# input_tensor = transform(image).unsqueeze(0)

# # Get model output
# output = model(input_tensor)

# # print((output.logits.detach().numpy()))

# # exit()
# # Convert output to tensor and apply squeeze method
# output_tensor = output.logits.argmax(dim=1).squeeze()
# tensor= output_tensor
# # Size_tensor = tf.size(output_tensor)
# print(output_tensor)


# # if tensor.ndim == 2:
# #     # grayscale image should have shape (height, width)
# #     tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], 1)
# # elif tensor.ndim == 3:
# #     # color image should have shape (height, width, channels)
# #     pass # no need to reshape
# # else:
# #     raise ValueError("Invalid tensor shape")

# print(output.logits.detach())


# tensor = output.logits.detach()

# # print(type(tensor))
# # pil_image = Image.fromarray(tensor)

# # # save the PIL image to a file
# # pil_image.save("numpy_image.png")



# output=tensor
# output = output.cpu().numpy()
# # cv.imwrite(output, 'pic.png')
# cv.imwrite('pic2.png', output)
# # array = tensor.numpy()
# exit()








# create PIL Image object from numpy array
# image = Image.fromarray(array)
# image.save('output_image.jpg')
# print(Size_tensor)
# plt.imshow(output_tensor.numpy()[0], cmap='gray')
# Convert tensor to PIL Image and save
# output_image = transforms.ToPILImage()(output_tensor)
# output_image.save('output_image.jpg')

# import torch
# import torchvision.transforms as transforms
# from PIL import Image

# # Load the model from the .pt file
# # model = torch.load('latest_model.pt')
# model = torch.load('latest_model.pt', map_location=torch.device('cpu'))


# # Set the model to evaluation mode
# model.eval()

# # Load the input image
# input_image = Image.open('A06_00Ab.png')

# # Define the image transformations
# image_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# # Apply the transformations to the input image
# input_tensor = image_transforms(input_image)

# # Add a batch dimension to the input tensor
# input_tensor = input_tensor.unsqueeze(0)

# # Pass the input tensor through the model
# output_tensor = model(input_tensor)

# # Remove the batch dimension from the output tensor and convert it to a PIL image
# output_image = transforms.ToPILImage()(output_tensor.squeeze())

# # Display the output image
# output_image.show()

# # Save the output image
# output_image.save('output_image.png')


# import torch
# import torchvision.transforms as transforms
# from PIL import Image

# from transformers import pipeline

# # Load the model
# model = pipeline("semantic-segmentation", model="flax-community/deit-base-384")

# # Define the input and output tensor shapes
# input_shape = (3, 384, 384)
# output_shape = (384, 384)

# # Define the transformation to apply to the input image
# transform = transforms.Compose([
#     transforms.Resize((input_shape[1], input_shape[2])),
#     transforms.ToTensor(),
# ])

# # Load the input image
# image = Image.open('input.jpg')

# # Apply the transformation to the input image
# input_tensor = transform(image)

# # Run the input tensor through the model
# output = model(image)

# # Extract the segmentation mask tensor from the output
# output_tensor = output['pixel_mask'].squeeze()

# # Convert the output tensor to an image
# output_image = transforms.ToPILImage()(output_tensor)

# # Save the output image
# output_image.save('output.jpg')








# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import torch
# import torchvision.transforms as transforms
# from PIL import Image

# # Load the PyTorch model onto the CPU
# model = torch.load('latest_model.pt', map_location=torch.device('cpu'))

# # Define the input and output tensor shapes
# input_shape = (3, 224, 224)
# output_shape = (3, 224, 224)

# # Define the transformation to apply to the input image
# transform = transforms.Compose([
#     transforms.Resize((input_shape[1], input_shape[2])),
#     transforms.ToTensor(),
# ])

# # Load the input image
# image = Image.open('A06_00Ab.png')

# # Apply the transformation to the input image
# input_tensor = transform(image).unsqueeze(0)

# # Run the input tensor through the model
# output_tensor = model(input_tensor)

# print(output_tensor)


# output=
# print((output.logits.detach().numpy()))
# # output=tensor
# output = output.cpu().numpy()
# # cv.imwrite(output, 'pic.png')
# cv.imwrite('pic1.png', output)
# # array = tensor.numpy()
# exit()


# output_image = transforms.ToPILImage()(output_tensor.squeeze())
# output_image.save('output.jpg')

# # # # Convert the output tensor to an image
# # # # output_image = transforms.ToPILImage()(output_tensor.squeeze(0))

# # # # print(output_tensor['pixel_mask'])
# # # # output_image = output_tensor['pixel_mask'].squeeze()
# # # # output_image.save('output.jpg')
# # # # Save the output image
# # # # output_image.save('output.jpg')

# # import torch
# # import torchvision.transforms as transforms
# # from PIL import Image

# # from transformers import pipeline
# # from torchvision import transforms

# # # Load the model
# # # model = pipeline("semantic-segmentation", model="latest_model.pt")
# # model = torch.load('latest_model.pt', map_location=torch.device('cpu'))

# # # Define the input and output tensor shapes
# # input_shape = (3, 224, 224)
# # output_shape = (224, 224)

# # # Define the transformation to apply to the input image
# # # transform = transforms.Compose([
# # #     transforms.Resize((input_shape[1], input_shape[2])),
# # #     transforms.ToTensor(),
# # # ])

# # transform = transforms.Compose([
# #     transforms.Resize((input_shape[1], input_shape[2])),
# #     transforms.ToTensor(),
# #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# # ])

# # # Load the input image
# # image = Image.open('A06_00Ab.png')

# # # # Apply the transformation to the input image
# # # input_tensor = transform(image)

# # # # Run the input tensor through the model
# # # output = model(image)
# # input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
# # output = model(input_tensor)

# # # Extract the segmentation mask tensor from the output
# # output_tensor = output.pixel_mask

# # # Convert the output tensor to an image
# # output_image = transforms.ToPILImage()(output_tensor)

# # # Save the output image
# # output_image.save('output.jpg')


# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import cv2 as cv

# # Load model
# model = torch.load('latest_model.pt', map_location=torch.device('cpu'))

# image = Image.open('A06_00Aa.png')

# input_shape = (3, 384, 384)

# transform = transforms.Compose([
#     transforms.Resize((input_shape[1], input_shape[2])),
#     transforms.ToTensor(),
# ])

# input_tensor = transform(image).unsqueeze(0)

# # Get model output
# output = model(input_tensor)

# # Convert output to tensor and apply squeeze method
# output_tensor = output.logits.argmax(dim=1).squeeze()
# tensor= (output.logits.detach())

# # Resize tensor to (3, 384, 384)
# resized_tensor = torch.nn.functional.interpolate(tensor, size=(input_shape[1], input_shape[2]))

# print(resized_tensor.ndim)
# print(type(resized_tensor))


# # Convert tensor to numpy array and then to image
# output_image = resized_tensor.cpu().numpy().astype(np.uint8)
# output_image = np.transpose(output_image, (1, 2, 0))

# # Convert numpy array to image
# output_image = Image.fromarray(output_image)

# # Save image
# output_image.save('pic2.png')


# Convert tensor to numpy array and then to image
# output_image = resized_tensor.cpu().numpy().astype(np.uint8)
# output_image = Image.fromarray(output_image.transpose((1, 2, 0)))

# # Save image
# cv.imwrite('pic2.png', output_image)

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2 as cv

# Load the pre-trained model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Set the device to use
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define the transform to be applied to the input image
transform = transforms.Compose([
    transforms.ToTensor()
])

# Read the input image
img = cv.imread('in.jpg')

# Convert the image from OpenCV format to PyTorch tensor format
img_tensor = transform(img).to(device)

# Make a prediction on the input image
model.eval()
with torch.no_grad():
    prediction = model([img_tensor])

# Get the predicted boxes and labels
boxes = prediction[0]['boxes'].cpu().numpy()
labels = prediction[0]['labels'].cpu().numpy()

# Draw rectangles around the detected objects
for i in range(len(boxes)):
    x1, y1, x2, y2 = boxes[i]
    label = labels[i]
    if label == 1:  # Class ID 1 corresponds to 'person'
        cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

cv.imwrite('pic2.png', img)


# # Display the output image
# cv2.imshow('Output Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
