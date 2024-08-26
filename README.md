# AI-Enhanced Subtyping of Thymic Tumors: Attention-based MIL with Pathology-Specific feature extraction
# Methodology
![flowChart](https://github.com/user-attachments/assets/054f6e7d-ef1b-4b3a-984c-afde38f5b197)

## Collection and Selection of WSIs from TCGA repositories:
Histopathology WSIs of 242 thymic epithelial tumors were obtained from the publicly accessible TCGA database (https://www.cancer.gov/tcga). These WSIs varied in size, ranging from 51 MB to 3.4 GB. The WSIs were categorized into six distinct subtypes/classes, with slide-level labels provided.
## Patch Extraction:
A customized [Yottixel](https://github.com/KimiaLabMayo/yottixel) script, originally developed by KimiaLab, was adopted to extract patches from tissue areas in the WSIs.
```
def RGB2HSD(X):
    eps = np.finfo(float).eps
    X[np.where(X==0.0)] = eps
    
    OD = -np.log(X / 1.0)
    D  = np.mean(OD,3)
    D[np.where(D==0.0)] = eps
    
    cx = OD[:,:,:,0] / (D) - 1.0
    cy = (OD[:,:,:,1]-OD[:,:,:,2]) / (np.sqrt(3.0)*D)
    
    D = np.expand_dims(D,3)
    cx = np.expand_dims(cx,3)
    cy = np.expand_dims(cy,3)
            
    X_HSD = np.concatenate((D,cx,cy),3)
    return X_HSD

def clean_thumbnail(thumbnail):
    thumbnail_arr = np.asarray(thumbnail)
    
    # writable thumbnail
    wthumbnail = np.zeros_like(thumbnail_arr)
    wthumbnail[:, :, :] = thumbnail_arr[:, :, :]

    # This  section sets regions with white spectrum as the background region
    thumbnail_std = np.std(wthumbnail, axis=2)
    wthumbnail[thumbnail_std<5] = (np.ones((1,3), dtype="uint8")*255)
    thumbnail_HSD = RGB2HSD( np.array([wthumbnail.astype('float32')/255.]) )[0]
    kernel = np.ones((30,30),np.float32)/900
    thumbnail_HSD_mean = cv2.filter2D(thumbnail_HSD[:,:,2],-1,kernel)
    wthumbnail[thumbnail_HSD_mean<0.05] = (np.ones((1,3),dtype="uint8")*255)
    return wthumbnail

def get_tissue_mask(thumbnail):
    cthumbnail = clean_thumbnail(thumbnail)
    tissue_mask = (cthumbnail.mean(axis=2) != 255) * 1
    return tissue_mask

def get_valid_patches(slide, thumbnail, tissue_threshold):
    tissue_mask = get_tissue_mask(thumbnail)
    w, h = slide.dimensions

    # at 20x its 1000x1000
    patch_size = 1000
    mask_hratio = (tissue_mask.shape[0] / h) * patch_size
    mask_wratio = (tissue_mask.shape[1] / w) * patch_size

    # iterating over patches
    patches = []

    for i, hi in enumerate(range(0, h, int(patch_size))):
        _patches = []
        for j, wi in enumerate(range(0, w, int(patch_size))):
            # check if patch contains 70% tissue area
            mi = int(i * mask_hratio)
            mj = int(j * mask_wratio)

            patch_mask = tissue_mask[mi:mi + int(mask_hratio), mj:mj + int(mask_wratio)]

            tissue_coverage = np.count_nonzero(patch_mask) / patch_mask.size

            _patches.append({'loc': [i, j], 'wsi_loc': [int(hi), int(wi)], 'tissue_coverage': tissue_coverage})

        patches.append(_patches)
    # for patch to be considered it should have this much tissue area
    valid_patches = []
    flat_patches = np.ravel(patches)
    for patch in tqdm.tqdm(flat_patches):
        # ignore patches with less tissue coverage
        if patch['tissue_coverage'] < tissue_threshold:
            continue

        h, w = patch['wsi_loc']
        patch_size_x = 250
        patch_region = slide.read_region((w, h), 0, (patch_size_x, patch_size_x)).convert('RGB')

        valid_patches.append(patch_region)
    print('valid patches:', len(valid_patches))
    return valid_patches
```

## Feature Extraction:
The following script has been developed to extract patches from WSIs, using the above Yottixel function, then to extract features from the produced patches on the fly (withous saving).
```
wsi_folder = "svs"
features = []
labels = []
wsi_ids = []
class_folders = sorted(os.listdir(wsi_folder))
class_ = class_folders[i]
class_path = os.path.join(wsi_folder, class_)
for wsi_id in os.listdir(class_path):
    # if wsi_id[:-4] in bags_wsi_ids:
    #     continue
    wsi_path = os.path.join(class_path, wsi_id)
    slide = openslide.open_slide(wsi_path)
    thumbnail = slide.get_thumbnail((500, 500))
    valid_patches =get_valid_patches(slide, thumbnail, 0.9)
    wsi_features = extract_features(valid_patches, 64)
    features.append(wsi_features)
    labels.append(class_)
    wsi_ids.append(wsi_id[:-4])
print(len(features), len(labels), len(wsi_ids))        
torch.save({'features': features, 'labels': labels, 'wsi_ids': wsi_ids}, f'{class_}_features_and_labels.pth')
```
Three different models were employed to extract features:
### ResNet50
```
data_transform  = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

import torch
import torch.nn as nn
from torchvision import models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define your ResNet50 model architecture here
model_rs = models.resnet50(weights='IMAGENET1K_V1')
numfeatures = model_rs.fc.in_features
model_rs.fc = nn.Linear(numfeatures, 5)
# Load the pretrained weights
model_rs.load_state_dict(torch.load('trained_resnet50_model.pth', map_location=device))
model_rs = nn.Sequential(*list(model_rs.children())[:-1])

model_rs = model_rs.to(device)

def extract_features(patches, batch_size):
    wsi_features = []
    tensors = [data_transform(patch) for patch in patches]    
    data_loader = torch.utils.data.DataLoader(tensors, batch_size=batch_size)
    for batch in tqdm.tqdm(data_loader):
        with torch.no_grad():
            batch = batch.to(device)
            batch_features = model_rs(batch)
            batch_features = batch_features.view(batch_features.size(0), -1)
            wsi_features.append(batch_features)
    wsi_features = torch.cat(wsi_features)
    print(wsi_features.shape)
    return wsi_features
```
### [HistoEncoder](https://github.com/jopo666/HistoEncoder)
```
data_transform  = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import histoencoder.functional as EF
encoder = EF.create_encoder("prostate_small")
encoder = encoder.to(device)

def extract_features(patches, batch_size):
    wsi_features = []
    tensors = [data_transform(patch) for patch in patches]    
    data_loader = torch.utils.data.DataLoader(tensors, batch_size=batch_size)
    for batch in tqdm.tqdm(data_loader):
        with torch.no_grad():
            batch = batch.to(device)
            batch_features = EF.extract_features(encoder, batch, num_blocks=2, avg_pool=True)
            wsi_features.append(batch_features)
    wsi_features = torch.cat(wsi_features)
    print(wsi_features.shape)
    return wsi_features
```
### [Phikon](https://github.com/owkin/HistoSSLscaling)
```
data_transform = transforms.Compose([transforms.ToTensor()])

from transformers import AutoImageProcessor, ViTModel

image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
phikon = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    phikon = nn.DataParallel(phikon)
phikon = phikon.to(device)

def extract_features(patches, batch_size):
    wsi_features = []
    tensors = [data_transform(patch) for patch in patches]    
    data_loader = torch.utils.data.DataLoader(tensors, batch_size=batch_size)
    for batch in tqdm.tqdm(data_loader):
        batch_processed = image_processor(batch, return_tensors="pt", do_rescale=False)
        with torch.no_grad():
            batch_processed.to(device)
            batch_outputs = phikon(**batch_processed)
            batch_features = batch_outputs.last_hidden_state[:, 0, :]
            wsi_features.append(batch_features)
    wsi_features = torch.cat(wsi_features)
    print(wsi_features.shape)
    return wsi_features
```

## Construction of Features Bags:
This self-developed approach involves grouping features into bags of uniform size, each containing 200 features/instances using (torch.chunk) function.
```
import torch

def process_data(input_paths):
    all_chunks = []
    all_labels = []
    chunk_size = 200
    for i, input_path in enumerate(input_paths):
        # Load data
        data = torch.load(input_path)
        features, labels = data['features'], data['labels']
        # Concatenate features
        features = torch.cat(features)
        # Calculate new size
        new_size = len(features) - (len(features) % chunk_size)
        print(new_size)
        # Slice features
        features = features[:new_size]
        # Split features into chunks
        chunks = torch.chunk(features, chunks=int(new_size / chunk_size), dim=0)
        # Generate labels
        labels = [i] * len(chunks)
        # Append to lists
        all_chunks.extend(chunks)
        all_labels.extend(labels)
    return all_chunks, all_labels
# List of input paths
input_paths = ['a_features_and_labels.pth', 'ab_features_and_labels.pth', 'b1_features_and_labels.pth',
               'b2_features_and_labels.pth', 'b3_features_and_labels.pth', 'ca_features_and_labels.pth']
# Process data
all_chunks, all_labels = process_data(input_paths)
# Save processed data
torch.save({'features': all_chunks, 'labels': all_labels}, 'tm_chunks200_yot_ds.pth')
```
## Classification Models:
AttenMIL, TransMIL, and Chowder were trained for classification.
```
coming soon
```
## Training Loop:
The training loop was implemented in PyTorch. It includes the following techniques: 
- ClassWeightedCrossEntropyLoss
- Adam Optimizer with learning rate scheduler
- Early Stopping
- Cross Validation
```
coming soon
```

For more information, see the original study: [10.1101/2024.06.07.24308609](https://doi.org/10.1101/2024.06.07.24308609 ).

If you wish to reuse any of the codes mentioned above, please ensure to cite the original manuscript accordingly.

```bibtex
@article {Kussaibi2024.06.07.24308609,
	author = {Kussaibi, Haitham},
	title = {AI-Enhanced Subtyping of Thymic Tumors: Attention-based MIL with Pathology-Specific Feature Extraction},
	elocation-id = {2024.06.07.24308609},
	year = {2024},
	doi = {10.1101/2024.06.07.24308609},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2024/08/17/2024.06.07.24308609},
	eprint = {https://www.medrxiv.org/content/early/2024/08/17/2024.06.07.24308609.full.pdf},
	journal = {medRxiv}
}
