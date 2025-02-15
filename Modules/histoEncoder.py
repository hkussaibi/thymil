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