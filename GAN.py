import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import glob 
import torch.nn.functional as F
import os
import numpy as np
import bdpy
from PIL import Image
import torchvision.models as models
from sklearn.linear_model import Ridge

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')




class Generator(nn.Module):
    def __init__(self, batch_size):
        super(Generator, self).__init__()
        self.batch_size = batch_size
        self.defc7 = nn.Linear(9919 + 9216, 4096)
        self.relu_defc7 = nn.ReLU(0.3)
        nn.init.kaiming_normal_(self.defc7.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.defc7.bias, 0)
        

        self.defc6 = nn.Linear(4096, 4096)
        self.relu_defc6 = nn.ReLU(0.3)
        nn.init.kaiming_normal_(self.defc6.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.defc6.bias, 0)

        self.defc5 = nn.Linear(4096, 4096)
        self.relu_defc5 = nn.ReLU(0.3)
        nn.init.kaiming_normal_(self.defc5.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.defc5.bias, 0)

        self.deconv5 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.relu_deconv5 = nn.ReLU(0.3)
        nn.init.kaiming_normal_(self.deconv5.weight, a=1.8, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.deconv5.bias, 0)
        
        

        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)  # Convolution
        self.relu_conv5_1 = nn.ReLU(0.3)
        nn.init.kaiming_normal_(self.conv5_1.weight, a=0.9, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv5_1.bias, 0)

        self.deconv4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # Deconvolution
        self.relu_deconv4 = nn.ReLU(0.3)
        nn.init.kaiming_normal_(self.deconv4.weight, a=1.8, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.deconv4.bias, 0)

        self.conv4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # Convolution
        self.relu_conv4_1 = nn.ReLU(0.3)
        nn.init.kaiming_normal_(self.conv4_1.weight, a=0.9, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv4_1.bias, 0)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # Deconvolution
        self.relu_deconv3 = nn.ReLU(0.3)
        nn.init.kaiming_normal_(self.deconv3.weight, a=1.8, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.deconv3.bias, 0)

        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)  # Convolution
        self.relu_conv3_1 = nn.ReLU(0.3)
        nn.init.kaiming_normal_(self.conv3_1.weight, a=0.9, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv3_1.bias, 0)

        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.relu_deconv2 = nn.ReLU(0.3)
        nn.init.kaiming_normal_(self.deconv2.weight, a=1.8, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.deconv2.bias, 0)

        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.relu_deconv1 = nn.ReLU(0.3)
        nn.init.kaiming_normal_(self.deconv1.weight, a=1.8, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.deconv1.bias, 0)

        self.deconv0 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        nn.init.kaiming_normal_(self.deconv0.weight, a=1.8, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.deconv0.bias, 0)

    def forward(self, x):
        
        x = self.defc7(x)
        x = self.relu_defc7(x)
        x = self.defc6(x)
        x = self.relu_defc6(x)
        x = self.defc5(x)
        x = self.relu_defc5(x)
        x = x.reshape(x.shape[0],256,4,4)
        x = self.deconv5(x)
        x = self.relu_deconv5(x)
        x = self.conv5_1(x)
        x = self.relu_conv5_1(x)
        x = self.deconv4(x)
        x = self.relu_deconv4(x)
        x = self.conv4_1(x)
        x = self.relu_conv4_1(x)
        x = self.deconv3(x)
        x = self.relu_deconv3(x)
        x = self.conv3_1(x)
        x = self.relu_conv3_1(x)
        x = self.deconv2(x)
        x = self.relu_deconv2(x)
        x = self.deconv1(x)
        x = self.relu_deconv1(x)
        x = self.deconv0(x)
        _, _, height, width = x.size()
        start_h = (height - 227) // 2
        start_w = (width - 227) // 2
        cropped_x = torch.narrow(x, 2, start_h, 227)
        cropped_x = torch.narrow(cropped_x, 3, start_w, 227)
        
        return cropped_x
class Discriminator(nn.Module):
    def __init__(self, batch_size):
        super(Discriminator, self).__init__()
        self.batch_size = batch_size
        self.Dconv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=4)
        nn.init.kaiming_normal_(self.Dconv1.weight, a=0.1, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.Dconv1.bias, 0)
        self.Drelu1 = nn.ReLU()

        self.Dconv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1)
        nn.init.kaiming_normal_(self.Dconv2.weight, a=1, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.Dconv2.bias, 0)
        self.Drelu2 = nn.ReLU()

        self.Dconv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        nn.init.kaiming_normal_(self.Dconv3.weight, a=1, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.Dconv1.bias, 0)
        self.Drelu3 = nn.ReLU()

        self.Dconv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        nn.init.kaiming_normal_(self.Dconv4.weight, a=1, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.Dconv4.bias, 0)
        self.Drelu4 = nn.ReLU()

        self.Dconv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2)
        nn.init.kaiming_normal_(self.Dconv5.weight, a=1, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.Dconv5.bias, 0)
        self.relu5 = nn.ReLU()

        self.Dpool5 = nn.AvgPool2d(kernel_size=11, stride=11)
        self.drop5 = nn.Dropout(p=0.5)

        self.Dfc6 = nn.Linear(256, 256)
        nn.init.kaiming_normal_(self.Dfc6.weight, a=0.1, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.Dfc6.bias, 0)
        self.Drelu6 = nn.ReLU()
        self.drop6 = nn.Dropout(p=0.5)

        self.Dfc7 = nn.Linear(256, 2)
        nn.init.kaiming_normal_(self.Dfc7.weight, a=0.1, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.Dfc7.bias, 0)
    def forward(self, x):
        x = self.Dconv1(x)
        x = self.Drelu1(x)
        x = self.Dconv2(x)
        x = self.Drelu2(x)
        x = self.Dconv3(x)
        x = self.Drelu3(x)
        x = self.Dconv4(x)
        x = self.Drelu4(x)
        x = self.Dconv5(x)
        x = self.relu5(x)
        x = self.Dpool5(x)
        x = x.reshape(x.shape[0],256)
        x = self.drop5(x)
        x = self.Dfc6(x)
        x = self.Drelu6(x)
        x = self.drop6(x)
        x = self.Dfc7(x)
        return x
class Comparator(nn.Module):
    def __init__(self):
        super(Comparator, self).__init__()
        alexnet = models.alexnet(pretrained=True)

        self.alexnet_without_fc = torch.nn.Sequential(*(list(alexnet.features.children())))

    def forward(self, x):   
        with torch.no_grad():
            self.alexnet_without_fc.eval()
            features = self.alexnet_without_fc(x)
            return features
fmri_data_table = [
    {'subject': 'sub-01',
     'data_file': '../data/fmri/sub-01_perceptionNaturalImageTraining_original_VC.h5',
     'roi_selector': 'ROI_VC = 1'},
    {'subject': 'sub-02',
     'data_file': '../data/fmri/sub-02_perceptionNaturalImageTraining_original_VC.h5',
     'roi_selector': 'ROI_VC = 1'},
    {'subject': 'sub-03',
     'data_file': '../data/fmri/sub-03_perceptionNaturalImageTraining_original_VC.h5',
     'roi_selector': 'ROI_VC = 1'}
]
fmri_data = np.load('./subj3_fmri.npy')
image_features = np.load('./image_features.npy')
image_dir = '../data/images/training'
image_file_pattern = '*.JPEG'
fmri_data_bd = bdpy.BData(fmri_data_table[2]['data_file'])
images_list = glob.glob(os.path.join(image_dir, image_file_pattern))  # List of image files (full path)
images_table = {os.path.splitext(os.path.basename(f))[0]: f
                    for f in images_list}
label_table = {os.path.splitext(os.path.basename(f))[0]: i + 1
                   for i, f in enumerate(images_list)}  
fmri_labels = fmri_data_bd.get('Label')[:, 1].flatten()
fmri_labels = ['n%08d_%d' % (int(('%f' % a).split('.')[0]),
                                 int(('%f' % a).split('.')[1]))
                   for a in fmri_labels]
class CustomDataset(Dataset):
    def __init__(self, fmri_data, image_features, frmi_labels, images_table):
        self.fmri_data = fmri_data
        self.image_features = image_features
        
        ridge_regressor = Ridge(alpha=1.0) 
        ridge_regressor.fit(fmri_data, image_features)

        self.pred_features = ridge_regressor.predict(fmri_data)
    def __len__(self):
        return len(fmri_data)
    def __getitem__(self, idx):
        image_path = images_table[fmri_labels[idx]]
        preprocess = transforms.Compose([
        transforms.Resize(248),
        transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path)
        image = np.asarray(image)
        if image.ndim == 2:
            img_rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype=image.dtype)
            img_rgb[:, :, 0] = image
            img_rgb[:, :, 1] = image
            img_rgb[:, :, 2] = image
            image = img_rgb
        input_tensor = preprocess(Image.fromarray(image))
        return torch.Tensor(fmri_data[idx]), torch.Tensor(image_features[idx]), torch.Tensor(self.pred_features[idx]),input_tensor
traindata = CustomDataset(fmri_data, image_features, fmri_labels, images_table)
image_weight = 100
feature_weight = 100
adversial_weight = 10000
batch_size = 100
generator = Generator(batch_size).to(device)
discriminator = Discriminator(batch_size).to(device)
comparator = Comparator().to(device)  

criterion = nn.CrossEntropyLoss() 
comparator_criterion = nn.MSELoss() 
image_criterion = nn.MSELoss() 
real_criterion = nn.CrossEntropyLoss() 
fake_criterion = nn.CrossEntropyLoss() 


gen_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.9, 0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.9, 0.999))



data_loader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
total_iterations = len(data_loader)
generator.to(device)
discriminator.to(device)
num_epochs = 200
train_discr = True
train_gen = True



for epoch in range(num_epochs):
    discriminators_loss = 0
    generators_loss = 0
    for i, (fmri_batch, feature_image_batch, pred_features, original_image_batch) in enumerate(data_loader):
        fmri_batch = fmri_batch.to(device)
        feature_image_batch = feature_image_batch.to(device)
        original_image_batch = original_image_batch.to(device)
        pred_features = pred_features.to(device)
        
        real_labels = torch.ones(batch_size, 1, dtype=torch.long).to(device).reshape(batch_size,)
        fake_labels = torch.zeros(batch_size, 1, dtype=torch.long).to(device).reshape(batch_size,)
        
        appended_fmri_batch = torch.concat((fmri_batch, pred_features), dim=1)
        discriminator.zero_grad()
        disc_optimizer.zero_grad()
        generated_images = generator(appended_fmri_batch)
        fake_discriminator = discriminator(generated_images)
        real_discriminator = discriminator(original_image_batch)
        discriminator_loss_real = real_criterion(real_discriminator, real_labels)
        discriminator_loss_fake = fake_criterion(fake_discriminator, fake_labels)
        discriminator_loss = discriminator_loss_fake + discriminator_loss_real
        if(train_discr):
            discriminator_loss.backward()
            disc_optimizer.step()
        else:
            discriminator_loss.backward()
            discriminator.zero_grad()
            disc_optimizer.zero_grad()
        comparator.zero_grad()
        comparator_features = comparator(generated_images)
    
        
        gen_optimizer.zero_grad()
        generator.zero_grad()
        generated_images2 = generator(appended_fmri_batch)
        real_labels = torch.ones(batch_size, 1, dtype=torch.long).to(device).reshape(batch_size,)
        fake_discriminator2 = discriminator(original_image_batch)
        feature_loss = comparator_criterion(comparator_features.reshape(batch_size,-1), feature_image_batch)
        image_loss = image_criterion(original_image_batch.reshape(batch_size,-1), generated_images2.reshape(batch_size,-1))
        adverisal_loss = criterion(fake_discriminator2, real_labels)
        # generator_loss = image_loss* image_weight + feature_loss*feature_weight + adverisal_loss*adversial_weight
        # generator_loss =feature_loss*feature_weight + adverisal_loss*adversial_weight
        # generator_loss = image_loss* image_weight  + adverisal_loss*adversial_weight
        generator_loss = image_loss* image_weight + feature_loss*feature_weight 
        if(train_gen):
            generator_loss.backward()
            gen_optimizer.step()
        else:
            generator_loss.backward()
            generator.zero_grad()
            gen_optimizer.zero_grad()
        discriminators_loss += discriminator_loss
        generators_loss += generator_loss
        discr_loss_ratio = discriminator_loss / generator_loss
        # if discr_loss_ratio < 1e-1 and train_discr:
        #   train_discr = False
        #   train_gen = True
        #   print("turned off discriminator")
        # if discr_loss_ratio > 5e-1 and not train_discr:
        #   train_discr = True
        #   train_gen = True
        #   print("turned on both")
        # if discr_loss_ratio > 1e1 and train_gen:
        #   train_gen = False
        #   train_discr = True
        #   print("turned off generator")
        
        
        
    if((epoch+1) % 10 == 0):
        torch.save(generator.to("cpu"), "generator_no_adverserial.pt")
        torch.save(discriminator.to("cpu"), "discriminator_no_adverserial.pt")
        print("saved")
        generator.to(device)
        discriminator.to(device)
    print(f'Epoch {epoch+1} Discriminator Loss: {discriminators_loss/total_iterations:}, Generator Loss: {generators_loss/total_iterations:} Feature Loss {feature_loss/total_iterations} Adverserial Loss {adverisal_loss/total_iterations} Image Loss {image_loss/total_iterations}')
torch.save(generator.to("cpu"), "generator_no_adverserial.pt")
torch.save(discriminator.to("cpu"), "discriminator_no_adverserial.pt")