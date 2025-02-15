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