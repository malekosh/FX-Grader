import nibabel as nib
import numpy as np
import os
import json
import nibabel.processing as nip
import nibabel.orientations as nio
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import PIL
from io import BytesIO
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Circle
import matplotlib as mpl
import warnings
from pathlib import Path
import pandas as pd
import math

mpl.rcParams['savefig.pad_inches'] = 0
v_dict = {
    1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7',
    8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6', 14: 'T7',
    15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12', 20: 'L1',
    21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'L6', 26: 'Sacrum',
    27: 'Cocc', 28: 'T13'
}

colors_itk = (1/255)*np.array([
    [255,  0,  0], [  0,255,  0], [  0,  0,255], [255,255,  0], [  0,255,255],
    [255,  0,255], [255,239,213],  # Label 1-7 (C1-7)
    [  0,  0,205], [205,133, 63], [210,180,140], [102,205,170], [  0,  0,128],
    [  0,139,139], [ 46,139, 87], [255,228,225], [106, 90,205], [221,160,221],
    [233,150,122], [165, 42, 42],  # Label 8-19 (T1-12)
    [255,250,250], [147,112,219], [218,112,214], [ 75,  0,130], [255,182,193],
    [ 60,179,113], [255,235,205],  # Label 20-26 (L1-6, sacrum)
    [255,235,205], [255,228,196],  # Label 27 cocc, 28 T13,
    [218,165, 32], [  0,128,128], [188,143,143], [255,105,180],  
    [255,  0,  0], [  0,255,  0], [  0,  0,255], [255,255,  0], [  0,255,255],
    [255,  0,255], [255,239,213],  # 29-39 unused
    [  0,  0,205], [205,133, 63], [210,180,140], [102,205,170], [  0,  0,128],
    [  0,139,139], [ 46,139, 87], [255,228,225], [106, 90,205], [221,160,221],
    [233,150,122],   # Label 40-50 (subregions)
    [255,250,250], [147,112,219], [218,112,214], [ 75,  0,130], [255,182,193],
    [ 60,179,113], [255,235,205], [255,105,180], [165, 42, 42], [188,143,143],
    [255,235,205], [255,228,196], [218,165, 32], [  0,128,128] # rest unused     
    ])
cm_itk = ListedColormap(colors_itk)
cm_itk.set_bad(color='w', alpha=0)

wdw_sbone = Normalize(vmin=-500, vmax=1300, clip=True)
wdw_hbone = Normalize(vmin=-200, vmax=1000, clip=True)

def reorient_to(img, axcodes_to=('P', 'I', 'R'), verb=False):
    
    # Note: nibabel axes codes describe the direction not origin of axes
    # direction PIR+ = origin ASL
    aff = img.affine
    arr = np.asanyarray(img.dataobj, dtype=img.dataobj.dtype)
    ornt_fr = nio.io_orientation(aff)
    ornt_to = nio.axcodes2ornt(axcodes_to)
    ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)
    arr = nio.apply_orientation(arr, ornt_trans)
    aff_trans = nio.inv_ornt_aff(ornt_trans, arr.shape)
    newaff = np.matmul(aff, aff_trans)
    newimg = nib.Nifti1Image(arr, newaff)
    if verb:
        print("[*] Image reoriented from", nio.ornt2axcodes(ornt_fr), "to", axcodes_to)
    return newimg

def crop_all(img, msk, ctd):
    ex_slc, o_shift = crop_slice(msk, dist=2)
    ctd = crop_centroids(ctd, o_shift)
    img = img.slicer[ex_slc]
    msk = msk.slicer[ex_slc]
        
    return img, msk, ctd

def crop_slice(msk, dist=10):
    shp = msk.dataobj.shape
    zms = msk.header.get_zooms()
    d = np.around(dist / np.asarray(zms)).astype(int)
    msk_bin = np.asanyarray(msk.dataobj, dtype=bool)
    msk_bin[np.isnan(msk_bin)] = 0
    cor_msk = np.where(msk_bin > 0)
    c_min = [cor_msk[0].min(), cor_msk[1].min(), cor_msk[2].min()]
    c_max = [cor_msk[0].max(), cor_msk[1].max(), cor_msk[2].max()]
    x0 = c_min[0]-d[0] if (c_min[0]-d[0]) > 0 else 0
    y0 = c_min[1]-d[1] if (c_min[1]-d[1]) > 0 else 0
    z0 = c_min[2]-d[2] if (c_min[2]-d[2]) > 0 else 0
    x1 = c_max[0]+d[0] if (c_max[0]+d[0]) < shp[0] else shp[0]
    y1 = c_max[1]+d[1] if (c_max[1]+d[1]) < shp[1] else shp[1]
    z1 = c_max[2]+d[2] if (c_max[2]+d[2]) < shp[2] else shp[2]
    ex_slice = tuple([slice(x0, x1), slice(y0, y1), slice(z0, z1)])
    origin_shift = tuple([x0, y0, z0])
    return ex_slice, origin_shift


def crop_centroids(ctd_list, o_shift):
    for v in ctd_list[1:]:
        v[1] = v[1] - o_shift[0]
        v[2] = v[2] - o_shift[1]
        v[3] = v[3] - o_shift[2]
    return ctd_list

def create_figure_1(dpi, planes, size=None):
    fig_h = round(2 * planes.shape[0] / dpi, 2)
    fig_w = round(2 * planes.shape[1] / dpi, 2)
    if size:
        s = size
    else:
        s=(fig_w, fig_h)
    fig, axs = plt.subplots(1, 1,figsize=s )
    axs.axis('off')
    return fig, axs, (fig_w,fig_h)

def process_data_snp(img_pth, msk_pth, ctd_pth):
    img = nib.load(img_pth)
    msk = nib.load(msk_pth)
    ctd_list = load_centroids(ctd_pth)
    
    img, msk, ctd_list = crop_all(img, msk, ctd_list)
    to_ax = ('I', 'A', 'L')
    img_iso = reorient_to(img, axcodes_to=to_ax)
    msk_iso = reorient_to(msk, axcodes_to=to_ax)
    ctd_iso = reorient_centroids_to(ctd_list, img_iso)
    
    drr_iso = img_iso.get_fdata().copy()
    drr_iso[msk_iso.get_fdata()==0] = np.nan
    
    drr = np.nansum(drr_iso,2)
    
    zms = img_iso.header.get_zooms()
    
    bsag_img_fx, bcor_img_fx, _, _, _ = sag_cor_curveprojection(ctd_iso, img_iso.get_fdata())
    
    bsag_img_fx = make_isotropic2d(bsag_img_fx, (zms[0], zms[1]))
    bcor_img_fx = make_isotropic2d(bcor_img_fx, (zms[0], zms[2]))
    drr = make_isotropic2d(drr, (zms[0], zms[1]))
    
    fig, axs, size = create_figure_1(120,bsag_img_fx,None)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    axs.imshow(bsag_img_fx, cmap=plt.cm.gray, norm=wdw_sbone)
    plot_sag_centroids(axs, ctd_iso, zms)
    buffer_ = BytesIO()
    plt.savefig(buffer_, format = "png")
    buffer_.seek(0)
    image = PIL.Image.open(buffer_)
    bsag_img_fx = np.asarray(image)
    buffer_.close()
    plt.close()
    
    fig, axs,_ = create_figure_1(120,bcor_img_fx,size)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    axs.imshow(bcor_img_fx, cmap=plt.cm.gray, norm=wdw_sbone)
    plot_cor_centroids(axs, ctd_iso, zms)
    buffer_ = BytesIO()
    plt.savefig(buffer_, format = "png")
    buffer_.seek(0)
    image = PIL.Image.open(buffer_)
    bcor_img_fx = np.asarray(image)
    buffer_.close()
    plt.close()

    return img_iso, ctd_iso, drr, bsag_img_fx, bcor_img_fx, zms, size
    


def make_isotropic2d(arr2d, zms2d, msk=False):
    xs = [x for x in range(arr2d.shape[0])]
    ys = [y for y in range(arr2d.shape[1])]
    if msk:
        interpolator = RegularGridInterpolator((xs, ys), arr2d, method='nearest')
    else:
        interpolator = RegularGridInterpolator((xs, ys), arr2d)
    new_shp = tuple(np.rint(np.multiply(arr2d.shape, zms2d)).astype(int))
    x_mm = np.linspace(0, arr2d.shape[0]-1, num=new_shp[0])
    y_mm = np.linspace(0, arr2d.shape[1]-1, num=new_shp[1])
    xx, yy = np.meshgrid(x_mm, y_mm)
    pts = np.vstack([xx.ravel(), yy.ravel()]).T
    img = np.reshape(interpolator(pts), new_shp, order='F')
    return img


def sag_cor_curveprojection(ctd_list, arr):
    # Sagittal and coronal projections of a curved plane defined by centroids
    # Note: Pass centoids (ctd_list) and image (img) in IAL direction!
    # if x-direction (=S/I) is not fully incremental, a straight, not an interpolated plane will be returned
    new_ctd=[]              # handle T13 = label 28
    [new_ctd.append([19.5,x[1],x[2],x[3]]) if x[0]==28 else new_ctd.append(x) for x in ctd_list[1:]]
    new_ctd.sort()
    out =[ctd_list[0]]
    [out.append(x) for x in new_ctd[1:]]
    ctd_arr = np.transpose(np.asarray(out[1:]))[1:]
    shp = arr.shape
    x_ctd = np.rint(ctd_arr[0]).astype(int)
    y_ctd = np.rint(ctd_arr[1]).astype(int)
    z_ctd = np.rint(ctd_arr[2]).astype(int)
    cor_plane = np.zeros((shp[0], shp[2]))
    sag_plane = np.zeros((shp[0], shp[1]))
    x_ctd_unique = np.unique(x_ctd)
    if (len(ctd_list) <= 3) or ((not np.array_equal(x_ctd, sorted(x_ctd_unique))) and (not  np.array_equal(x_ctd,sorted(x_ctd_unique, reverse=True)))):
        y_cord = np.rint(np.mean(y_ctd)).astype(int)
        z_cord = np.rint(np.mean(z_ctd)).astype(int)
        cor_plane = arr[:, y_cord, :]
        sag_plane = arr[:, :, z_cord]
        y_cord = np.array([y_cord]*shp[1])
        z_cord = np.array([z_cord]*shp[2])
    else:
        f_sag = interp1d(x_ctd, y_ctd, kind='quadratic')
        f_cor = interp1d(x_ctd, z_ctd, kind='quadratic')
        window_size, poly_order = int((max(x_ctd)-min(x_ctd))/2), 3
        if  window_size % 2 == 0:
            window_size+=1
        y_cord = np.array([np.rint(f_sag(x)).astype(int) for x in range(min(x_ctd), max(x_ctd))])
        y_cord = np.rint(savgol_filter(y_cord, window_size, poly_order)).astype(int)
        z_cord = np.array([np.rint(f_cor(x)).astype(int) for x in range(min(x_ctd), max(x_ctd))])
        z_cord = np.rint(savgol_filter(z_cord, window_size, poly_order)).astype(int)
        y_cord[y_cord <  0]= 0                      #handle out-of-volume interpolations
        y_cord[y_cord >= shp[1]] = shp[1] - 1
        z_cord[z_cord < 0] = 0
        z_cord[z_cord >= shp[2]] = shp[2] - 1      
        for x in range(0, shp[0] - 1):
            if x < min(x_ctd):
                cor_plane[x, :] = arr[x, y_cord[0], :]
                sag_plane[x, :] = arr[x, :, z_cord[0]]
            elif x >= max(x_ctd):
                cor_plane[x, :] = arr[x, y_cord[-1], :]
                sag_plane[x, :] = arr[x, :, z_cord[-1]]
            else:
                cor_plane[x, :] = arr[x, y_cord[x - min(x_ctd)], :]
                sag_plane[x, :] = arr[x, :, z_cord[x - min(x_ctd)]]
    return sag_plane, cor_plane, y_cord, z_cord, x_ctd

def reorient_centroids_to(ctd_list, img, decimals=1, verb=False):
    # reorient centroids to image orientation
    # todo: reorient to given axcodes (careful if img ornt != ctd ornt)
    ctd_arr = np.transpose(np.asarray(ctd_list[1:]))
    if len(ctd_arr) == 0:
        print("[#] No centroids present") 
        return ctd_list
    v_list = ctd_arr[0].astype(int).tolist()  # vertebral labels
    ctd_arr = ctd_arr[1:]
    ornt_fr = nio.axcodes2ornt(ctd_list[0])  # original centroid orientation
    axcodes_to = nio.aff2axcodes(img.affine)
    ornt_to = nio.axcodes2ornt(axcodes_to)
    trans = nio.ornt_transform(ornt_fr, ornt_to).astype(int)
    perm = trans[:, 0].tolist()
    shp = np.asarray(img.dataobj.shape)
    ctd_arr[perm] = ctd_arr.copy()
    for ax in trans:
        if ax[1] == -1:
            size = shp[ax[0]]
            ctd_arr[ax[0]] = np.around(size - ctd_arr[ax[0]], decimals)
    out_list = [axcodes_to]
    ctd_list = np.transpose(ctd_arr).tolist()
    for v, ctd in zip(v_list, ctd_list):
        out_list.append([v] + ctd)
    if verb:
        print("[*] Centroids reoriented from", nio.ornt2axcodes(ornt_fr), "to", axcodes_to)
    return out_list

def load_centroids(ctd_path):
    with open(ctd_path) as json_data:
        dict_list = json.load(json_data)
        json_data.close()
    ctd_list = []
    for d in dict_list:
        if 'direction' in d:
            ctd_list.append(tuple(d['direction']))
        elif 'nan' in str(d):            #skipping NaN centroids
            continue
        else:
            ctd_list.append([d['label'], d['X'], d['Y'], d['Z']]) 
    return ctd_list



def create_figure(dpi, *planes):
    fig_h = round(2 * planes[0].shape[0] / dpi, 2)
    plane_w = [p.shape[1] for p in planes]
    w = sum(plane_w)
    fig_w = round(2 * w / dpi, 2)
    x_pos = [0]
    for x in plane_w[:-1]:
        x_pos.append(x_pos[-1] + x)
    fig, axs = plt.subplots(1, len(planes), figsize=(fig_w, fig_h))
    for a in axs:
        a.axis('off')
        idx = axs.tolist().index(a)
        a.set_position([x_pos[idx]/w, 0, plane_w[idx]/w, 1])
    return fig, axs


def plot_sag_centroids(axs, ctd, zms):
    # requires v_dict = dictionary of mask labels
    for v in ctd[1:]:
        axs.add_patch(Circle((v[2]*zms[1], v[1]*zms[0]), 1, color=colors_itk[v[0]-1]))
        axs.text(4, v[1]*zms[0], v_dict[v[0]], fontdict={'color': cm_itk(v[0]-1), 'weight': 'bold'})


def plot_cor_centroids(axs, ctd, zms):
    # requires v_dict = dictionary of mask labels
    for v in ctd[1:]:
        axs.add_patch(Circle((v[3]*zms[2], v[1]*zms[0]), 1, color=colors_itk[v[0]-1]))
        axs.text(4, v[1]*zms[0], v_dict[v[0]], fontdict={'color': cm_itk(v[0]-1), 'weight': 'bold'})
        
        
def save_json(out_dict, out_path):
    def convert(o):
        if isinstance(o, np.int64):
            return int(o)
        raise TypeError
    with open(out_path, 'w') as f:
        json.dump(out_dict, f, indent=4, default=convert)
    print("[*] Json saved:", out_path)
    

def sag_i(x,sag):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    fig = plt.figure(figsize=(3, 12))
    ax_neu = fig.add_subplot(1,1,1) 
    plt.axis('off')

    ax_neu.imshow(sag)
    fig.canvas.draw()
    
def sag_drr(x,drr):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    fig = plt.figure(figsize=(3, 12))
    ax_neu = fig.add_subplot(1,1,1) 
    plt.axis('off')

    ax_neu.imshow(drr,  cmap=plt.cm.gray, interpolation='lanczos')
    fig.canvas.draw()

def cor_i(x,cor):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    fig = plt.figure(figsize=(3, 12))
    ax_neu = fig.add_subplot(1,1,1) 
    plt.axis('off')

    ax_neu.imshow(cor)
    fig.canvas.draw()
    
def cor_f(slc,img,zms,size):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img_median = img.get_fdata()[:,slc,:]
        
    img_median = make_isotropic2d(img_median,(zms[0], zms[2]))
        
    fig, axs,_ = create_figure_1(120,img_median,size)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    axs.imshow(img_median, cmap=plt.cm.gray, norm=wdw_sbone)
    buffer_ = BytesIO()
    plt.savefig(buffer_, format = "png")
    buffer_.seek(0)
    image = PIL.Image.open(buffer_)
    img_median = np.asarray(image)
    buffer_.close()
    plt.close()    

    fig = plt.figure(figsize=(3, 12))
    ax_neu = fig.add_subplot(1,1,1) 
    plt.axis('off')

    ax_neu.imshow(img_median)
    fig.canvas.draw()
    
def sag_f(slc,img,zms):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img_median = img.get_fdata()[:,:,slc]
    fig = plt.figure(figsize=(3, 12))
    ax_neu = fig.add_subplot(1,1,1) 
    plt.axis('off')

    ax_neu.imshow(make_isotropic2d(img_median,(zms[0], zms[1])), cmap="gray", norm=wdw_sbone)
    fig.canvas.draw()
#     display(fig)

def ax(slc):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img_median = img.get_fdata()[slc,:,:]
    fig = plt.figure(figsize=(6, 5))
    ax_neu = fig.add_subplot(1,1,1) 
    plt.axis('off')

    ax_neu.imshow(img_median, cmap="gray")
    fig.canvas.draw()
    
    
v_dict = {
    2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7',
    8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6', 14: 'T7',
    15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12', 20: 'L1',
    21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'L6', 28: 'T13'
}

def get_paths(pth, ex, rater,chunks=None):
    
    im_paths = []
    study_dir = Path(pth)
    if study_dir.joinpath('rawdata').is_dir():
        subject_dirs = sorted(study_dir.joinpath('rawdata').iterdir())
    else:
        subject_dirs = sorted(study_dir.iterdir())
    #select only specific studies or skip specific studies
    #if not ("dataset-CSI" in str(study_dir)) or ("dataset-csi" in str(study_dir)):
    #    continue
    file_list=[]

    all_dicts = []
    if chunks is not None:
        subject_dirs = subject_dirs[(chunks*20):((chunks*20)+20)]
    for subject_dir in subject_dirs:                 # iterate over subjectdirs
        if not subject_dir.is_dir() or subject_dir.name[0] == '.':
            continue
        files = list(subject_dir.rglob('*_ct.nii.gz'))      # rglob will iterate over files indepentent from the presence of sessions
        if len(files) > 0:  # skip root_dirs and empty dirs
            files.sort()
        

        for f_pth in files:
 
            if "rawdata" in  str(f_pth):
                drv = Path(str(f_pth).replace("rawdata", "derivatives"))
                msk_pth= drv.with_name(drv.name.replace("_ct.nii.gz", "_seg-vert_msk.nii.gz"))
                ctd_pth = drv.with_name(drv.name.replace("_ct.nii.gz", "_seg-subreg_ctd.json"))
                res_path = drv.with_name(drv.name.replace("_ct.nii.gz", '_fx-{}_res.json'.format(rater)))
                if ex:
                    if res_path.is_file():
                        continue                
                if not msk_pth.is_file() or not ctd_pth.is_file():
                    continue
                else:
                    im_paths.append([str(f_pth), str(msk_pth),str(ctd_pth)])

                
    return im_paths


def save_csv(pth, rater):
    fx_path = Path(pth).joinpath('FX-Grades_{}.csv'.format(rater))
    ivd_path = Path(pth).joinpath('IVD-Grades_{}.csv'.format(rater))
    
    study_dir = Path(pth)
    if study_dir.joinpath('derivatives').is_dir():
        subject_dirs = sorted(study_dir.joinpath('derivatives').iterdir())
    else:
        subject_dirs = sorted(study_dir.iterdir())
    

    all_fx_dicts = []
    all_ivd_dicts = []
    files_fx = list(study_dir.rglob('*_fx-{}_res.json'.format(rater)))      
    files_ivd = list(study_dir.rglob('*_ivd-{}_res.json'.format(rater)))   

    for fx_pth in files_fx:
        all_fx_dicts.append(load_json(fx_pth))         
    for ivd_pth in files_ivd:
        all_ivd_dicts.append(load_json(ivd_pth)) 
    data_frame_fx  = pd.DataFrame(all_fx_dicts)
    data_frame_fx.to_csv(fx_path)
    print('# Fracture csv saved : ',fx_path)
    data_frame_ivd  = pd.DataFrame(all_ivd_dicts)
    data_frame_ivd.to_csv(ivd_path)   
    print('# IVD csv saved : ',ivd_path)

def load_json(json_path):
    name = Path(json_path).name
    name = str(name).split('_')[0]
    with open(json_path) as json_data:
        js = json.load(json_data)
        js['name'] = name
        json_data.close()
    return js


def get_result_jan(path,vert):
    try:
        grade_table = pd.read_excel(str(Path.cwd().joinpath('fx_jsk_verse.xlsx')))
        ID = int(str(Path(path).name).split('_')[0][-3:])
        gr = grade_table[vert][(grade_table["ID"] == int(ID))]
        if not gr.empty:
            if not math.isnan(gr):
                gr = str(int(gr))
            else:
                gr = ''
        else:
            gr = ''
        return gr
    except Exception as e:
        return ''
    
def get_result_thomas(path,vert):
    try:
        grade_table = pd.read_excel(str(Path.cwd().joinpath('fx_thomas_verse.xlsx')))
        ID = int(str(Path(path).name).split('_')[0][-3:])
        gr = grade_table[vert][(grade_table["ID"] == int(ID))]
        if not gr.empty:
            if not math.isnan(gr):
                gr = str(int(gr))
            else:
                gr = ''
        else:
            gr = ''
        return gr
    except Exception as e:
        return ''
    
def get_result_max(path,vert):
    try:
        if 'C' in vert:
            return ''
        grade_table = pd.read_excel(str(Path.cwd().joinpath('ryai190138_appendixe1.xlsx')),sheet_name= 'VerSe_dataset')
        ID = int(str(Path(path).name).split('_')[0][-3:])
        vert = vert + '_fx-s'
        gr = grade_table[vert][(grade_table["verse_ID"] == ID)]
        if not gr.empty:
            if not math.isnan(gr):
                gr = str(int(gr))
            else:
                gr = ''
        else:
            gr = ''
        return gr
    except Exception as e:
        return ''