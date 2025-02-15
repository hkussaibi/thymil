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