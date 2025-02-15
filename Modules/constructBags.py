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