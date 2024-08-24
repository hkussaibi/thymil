
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
