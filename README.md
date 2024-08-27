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
### [TransMIL](https://github.com/szc19990412/TransMIL)
### [Chowder](http://arxiv.org/pdf/1802.02212)
### AttenMIL: ©
```
def normalize_input_data(x, epsilon=1e-5):
    mean = x.mean()
    std = x.std(unbiased=False) + epsilon  # Adding epsilon to avoid division by zero
    normalized_x = (x - mean) / std
    return normalized_x

class MulticlassClassifier(nn.Module):
    def __init__(self, input_size=768, hidden_size=128, num_classes=6, dropout_rate=0.5):
        super(MulticlassClassifier, self).__init__()
        self.inst_norm_input = nn.InstanceNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size, 64)
        init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(64, num_classes)
        init.xavier_uniform_(self.fc3.weight, gain=nn.init.calculate_gain('relu'))
        
        self.attention_layer = nn.Linear(num_classes, 1)
        init.xavier_uniform_(self.attention_layer.weight, gain=nn.init.calculate_gain('relu'))
        
    def forward(self, x):
        x = normalize_input_data(x)
        x = self.inst_norm_input(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        instance_outputs = self.fc3(x)  # [batch_size, num_instances, num_classes]

        # Compute attention scores
        attention_scores = self.attention_layer(instance_outputs).squeeze(-1)
        attention_scores = F.softmax(attention_scores, dim=-1)

        # Compute bag-level prediction using attention-based pooling
        bag_output = torch.sum(instance_outputs * attention_scores.unsqueeze(-1), dim=1)

        return bag_output
```
## Training Loop:
The training loop was implemented in PyTorch.
```
# Create a directory to save models for each fold
os.makedirs(f'{extractor}/{classifier_name}/', exist_ok=True)
output_dir = f'{extractor}/{classifier_name}/'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler(f'{output_dir}training.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
```
Class Weighted Cross Entropy Loss
```
class ClassWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_sizes, device='cuda', reduction='mean', ignore_index=-100):
        super(ClassWeightedCrossEntropyLoss, self).__init__()
        total_samples = sum(class_sizes)
        class_weights = [total_samples / (len(class_sizes) * size) for size in class_sizes]
        self.weight = torch.tensor(class_weights, dtype=torch.float32).to(device)  # Move tensor to device
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        input = input  # Assuming input is a tuple (logits, ...)
        loss = nn.functional.cross_entropy(input, target, weight=self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss
```
```
class SimpleClassifierTraining:
    def __init__(self, classifier, train_data_loader, val_data_loader, class_sizes, device='cuda'):
        self.classifier = classifier
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.device = device
        self.class_sizes = class_sizes

        # Define loss function and optimizer
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = ClassWeightedCrossEntropyLoss(class_sizes=class_sizes)
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=0.001, weight_decay=0.001)

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', patience=2, verbose=True)

    def train(self, num_epochs):
        patience = 10
        epochs_no_improve = 0
        best_model_state = None
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            # Training phase
            self.classifier.train()
            total_loss = 0.0
            correct_train = 0
            total_train = 0

            for batch_features, batch_labels in tqdm.tqdm(self.train_data_loader):
                batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.classifier(batch_features)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()

                # Calculate accuracy
                predicted_train = outputs.max(1)[1]
                total_train += batch_labels.size(0)
                correct_train += (predicted_train == batch_labels).sum().item()

            average_train_loss = total_loss / len(self.train_data_loader)
            train_accuracy = correct_train / total_train

            # Validation phase
            self.classifier.eval()
            total_val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for val_batch_features, val_batch_labels in tqdm.tqdm(self.val_data_loader):
                    val_batch_features, val_batch_labels = val_batch_features.to(self.device), val_batch_labels.to(self.device)
                    val_outputs = self.classifier(val_batch_features)
                    val_loss = self.criterion(val_outputs, val_batch_labels)
                    total_val_loss += val_loss.item()

                    # Calculate validation accuracy
                    predicted_val = val_outputs.max(1)[1]
                    total_val += val_batch_labels.size(0)
                    correct_val += (predicted_val == val_batch_labels).sum().item()

            average_val_loss = total_val_loss / len(self.val_data_loader)
            val_accuracy = correct_val / total_val

            # Print training and validation statistics
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {average_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            # Learning rate scheduler step
            self.scheduler.step(val_accuracy)

            # Check for early stopping
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                epochs_no_improve = 0
                best_model_state = self.classifier.state_dict()  # Save the best model state
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    logger.info("Early stopping triggered!")
                    break

        # Save the best model state after completing training for each fold
        torch.save(best_model_state, f"{output_dir}fold_{fold}_best_model.pth")
        print(f"Best model weights for fold {fold} saved.")
```
Cross Validation
```

# Specify the number of folds
num_folds = 5  # Adjust as needed
batch_size = 32  # Adjust as needed
dataset = train_data

# Create KFold object
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Iterate over folds
for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    logger.info(f"Fold {fold + 1}/{num_folds}")
    # Split your dataset into training and validation sets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    # Create data loaders for this fold
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # With class sizes
    trainer = SimpleClassifierTraining(classifier, train_data_loader, val_data_loader, class_sizes, device='cuda')

    # Train and evaluate your model for this fold
    trainer.train(num_epochs=100)
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
```
© 2024 anapath.org This code is made available under the Apache-2 License and is available for non-commercial academic purposes.
