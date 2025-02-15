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