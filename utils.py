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
import time
from copy import deepcopy
from skimage.transform import resize

mpl.rcParams['savefig.pad_inches'] = 0
v_dict = {1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7',
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

cm_sr = (1/255)*np.array([
        [104,159,56], [ 216,67,28] 
        ])
    

cm_sr = ListedColormap(cm_sr)
cm_sr.set_bad(color='w', alpha=0)


cm_bmd_co = (1/255)*np.array([
        [0,0,0],[104,159,56], [255,179,0], [ 216,67,28], [183,28,28], # green, yellow, red, dark red for >120, >80, >50, <50
        ])


    

cm_bmd = ListedColormap(cm_bmd_co)
cm_bmd.set_bad(color='w', alpha=0)

colors_labels = (1/255)*np.array([
    [  0,  0,  0],  #   
    [230,255,  0],  #   "Label 1"
    [255,247,  0],  #   "Label 3"
    [255,162,  0],  #   "Label 5"
    [255,  0, 12],  #   "Label 4"
    [166,  0,255],  #   "Label 5"
    [  0,  0,255],  #   "Label 6"
    [ 20,108,  0],  #   "Label 7"
    ])


cm_labl = ListedColormap(colors_labels)
cm_labl.set_bad(color='w', alpha=0)  # set NaN to full opacity for overlay

def plot_bmd_label(axs, ctd, zms, bmd_val, size=3, text=True):
    # requires v_dict = dictionary of mask labels and a pd dataframe with labels "ctd_labels"
    try:
        bmd_val = float(bmd_val)

        if bmd_val > 120:
            bmd = 1
        elif bmd_val > 80:
            bmd = 2
        elif bmd_val > 50:
            bmd = 3
        else:
            bmd = 4
#         print('bmd ', bmd)
        for v in ctd[1:]:
            if v[0] ==22:
                axs.add_patch(Circle((v[2]*zms[1], v[1]*zms[0]), size, color=cm_bmd_co[bmd]))
                if text:
                    axs.text(v[2]*zms[1]- 40, v[1]*zms[0]+2, str(bmd_val), fontdict={'color': cm_bmd(bmd), 'weight': 'bold'}, backgroundcolor='black')
    except Exception as e:
        pass
    
    
def initialize_database(rater):
    csv_path = 'CTFU_Fx_{}.csv'.format(rater)
    if not os.path.isfile(csv_path):
        columns = ['ID', 'type', 'ce', 
                   'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 
                   'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
                   'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'T13']
        df = pd.DataFrame(columns= columns)
        df.to_csv(csv_path, index=False)
    return csv_path
        
def populate_database(database_path, fx_dict, img_path, typ):
    print(database_path)
    df = pd.read_csv(database_path)
    fx_dict['ID'] = str(os.path.basename(img_path).replace('_ct.nii.gz', ''))
    if fx_dict['ID'] in df.ID.values:
        idx = df.index[df['ID']==fx_dict['ID']].tolist()
        df = df.drop(index=idx, axis=0)# = fx
    fx_dict['ce'] = get_ce(img_path)
    fx_dict['type'] = typ
    df = df.append(fx_dict, ignore_index=True)
    df.to_csv(database_path, index=False)
    return True

def get_ce(pth):
    try:
        ce = str(pth).split('ce-')[1].split('_')[0]
    except Exception:
        ce = ''
    return ce

def reorient_to(img, axcodes_to=('P', 'I', 'R'), verb=False, msk=True):
    
    # Note: nibabel axes codes describe the direction not origin of axes
    # direction PIR+ = origin ASL
    aff = img.affine
    if msk:
        arr = np.asanyarray(img.dataobj, dtype=img.dataobj.dtype)
    else:
        arr = img.get_fdata()
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


def get_paddings(msk1, msk2, ctd1, ctd2):

    rat_pad_1_1, rat_pad_1_2 = 0, 0
    rat_pad_2_1, rat_pad_2_2 = 0, 0
    
    ctd_dict1 = {x[0] : [x[1], x[2], x[3]] for x in ctd1[1:]}
    ctd_dict2 = {x[0] : [x[1], x[2], x[3]] for x in ctd2[1:]}

    if msk2.shape[1] > msk1.shape[1]:
        rat_pad_1_1 = len([x for x in ctd_dict2.keys() if x<min(list(ctd_dict1.keys()))])
        rat_pad_1_2 = len([x for x in ctd_dict2.keys() if x>max(list(ctd_dict1.keys()))])

        to_pad_tot = msk2.shape[1] - msk1.shape[1]
        rat_pad_1_1 = (rat_pad_1_1/(rat_pad_1_1 + rat_pad_1_2+1e-7)) * to_pad_tot
        rat_pad_1_2 = (rat_pad_1_2/(rat_pad_1_1 + rat_pad_1_2+1e-7)) * to_pad_tot

    if msk1.shape[1] > msk2.shape[1] > 0:
        to_pad_tot = msk1.shape[1] - msk2.shape[1]
        rat_pad_2_1 = len([x for x in ctd_dict1.keys() if x<min(list(ctd_dict2.keys()))])
        rat_pad_2_2 = len([x for x in ctd_dict1.keys() if x>max(list(ctd_dict2.keys()))])

        rat_pad_2_1 = (rat_pad_2_1/(rat_pad_2_1 + rat_pad_2_2+1e-7)) * to_pad_tot
        rat_pad_2_2 = (rat_pad_2_2/(rat_pad_2_1 + rat_pad_2_2+1e-7)) * to_pad_tot


    if msk2.shape[0] > msk1.shape[0]:
        to_pad_rl_1 = msk2.shape[0] - msk1.shape[0]
        to_pad_rl_2 = 0
    elif msk1.shape[0]>  msk2.shape[0]:
        to_pad_rl_2 = msk1.shape[0] - msk2.shape[0]
        to_pad_rl_1 = 0
    else:
        to_pad_rl_1,  to_pad_rl_2= 0, 0
        
    if msk2.shape[2] > msk1.shape[2]:
        to_pad_io_1 = msk2.shape[2] - msk1.shape[2]
        to_pad_io_2 = 0
    elif msk1.shape[2]>  msk2.shape[2]:
        to_pad_io_2 = msk1.shape[2] - msk2.shape[2]
        to_pad_io_1 = 0
    else:
        to_pad_io_1,  to_pad_io_2= 0, 0
        
    return int(rat_pad_1_1), int(rat_pad_1_2), int(rat_pad_2_1), int(rat_pad_2_2), int(to_pad_rl_1), int(to_pad_rl_2), int(to_pad_io_1),  int(to_pad_io_2)

def pad_arr(arr1, arr2, pads, msk=False):
    if not msk:
        cnst = (-1024)
    else:
        cnst = (0)
    
    new_arr1 = np.pad(
        arr1,
        [(0, int(pads[4])), (int(pads[0]), int(pads[1])), (0, int(pads[6]))],
        mode='constant',
        constant_values = cnst
    )
    new_arr2 = np.pad(
        arr2,
        [(0, pads[5]), (pads[2], pads[3]), (0, pads[7])],
        mode='constant',
        constant_values = cnst
    )
    return new_arr1, new_arr2

def pad_ctds(ctd1, ctd2, pads):
    new_ctd1 = [[x[0], x[1], x[2]+pads[0], x[3]] if isinstance(x,list) else x for x in  ctd1]
    
    new_ctd2 = [[x[0], x[1], x[2]+pads[2], x[3]] if isinstance(x,list) else x for x in  ctd2]
    return new_ctd1, new_ctd2

def pad_niftis(bimg, bmsk,sr, fimg, fmsk, res, bctd, fctd):
    bimg_data = bimg.get_fdata()
    bmsk_data = np.asanyarray(bmsk.dataobj, dtype=bmsk.dataobj.dtype)
    fimg_data = fimg.get_fdata()
    fmsk_data = np.asanyarray(fmsk.dataobj, dtype=fmsk.dataobj.dtype)
    


    res_data = np.asanyarray(res.dataobj, dtype=res.dataobj.dtype)
    sr_arr = np.asanyarray(sr.dataobj, dtype=sr.dataobj.dtype)
    
    pads = get_paddings(bmsk_data,fmsk_data,bctd,fctd)

    bimg_data, fimg_data =  pad_arr(bimg_data, fimg_data, pads, msk=False)

    bmsk_data, fmsk_data =  pad_arr(bmsk_data, fmsk_data, pads, msk=True)
    sr_arr, _ = pad_arr(sr_arr, sr_arr, pads, msk=True)
    res_data, _ = pad_arr(res_data, res_data, pads, msk=True)
    
    bctd1, fctd1 = pad_ctds(bctd, fctd, pads)
    
    bimg1 = nib.Nifti1Image(bimg_data,bimg.affine)
    bmsk1 = nib.Nifti1Image(bmsk_data,bmsk.affine)
    res1 = nib.Nifti1Image(res_data,res.affine)
    sr1 = nib.Nifti1Image(sr_arr,sr.affine)
    fimg1 = nib.Nifti1Image(fimg_data,fimg.affine)
    fmsk1 = nib.Nifti1Image(fmsk_data,fmsk.affine)
    return bimg1, bmsk1,sr1, fimg1, fmsk1, res1, bctd1, fctd1 


def get_labels_from_csv(dataframe, case_id, exclude_cervical=False):
    verts = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7', 
                   'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
                   'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'T13']
    all_entries = dataframe[dataframe['ID']==case_id].to_dict(orient='records')[0]
    v_bmd = all_entries.get('vBMD_L3')
    if np.isnan(v_bmd):
        v_bmd = ''
    
        
    return dataframe[dataframe['ID']==case_id], v_bmd

def get_labels_from_imp(dataframe, img_path):
    verts = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7', 
                   'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
                   'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'T13']
    
    case_id =  get_csv_entry(img_path)
    all_entries = dataframe[dataframe['ID']==case_id].to_dict(orient='records')[0]
    
    labels_dict = {x:y for x,y in all_entries.items() if 'def_' in x and not np.isnan(y)}

        
    return labels_dict

    

def process_data_snp_sub(rater_dict, labels, dpi=120):
    img_pth = rater_dict['b_img_pth']
    fimg_pth = rater_dict['f_img_pth']
    sub_path = rater_dict['sub_pth']
    # img_pth, msk_pth, sr_pth, ctd_pth,fimg_pth, fmsk_pth, fctd_pth, sub_path, labels, dpi=120
    msk_pth, sr_pth, ctd_pth = get_derivative_path_from_raw_ct(img_pth)
    fmsk_pth, fsr_pth, fctd_pth = get_derivative_path_from_raw_ct(fimg_pth)
    
    
    b_labels, bbmd = get_labels_from_csv(labels, get_csv_entry(img_pth))
    f_labels, fbmd = get_labels_from_csv(labels, get_csv_entry(fimg_pth))
    
    bimg = nib.load(img_pth)
    bmsk = nib.load(msk_pth)
    bctd = load_centroids(ctd_pth)
    sub = nib.load(sub_path)
    sr = nib.load(sr_pth)
    b_ex_cents = [x[0] for x in bctd[1:] if isinstance(x[0], int)]
    
    fimg = nib.load(fimg_pth)
    fmsk = nib.load(fmsk_pth)
    fctd = load_centroids(fctd_pth)
    f_ex_cents = [x[0] for x in fctd[1:] if isinstance(x[0], int)]
    
    to_remove_bs = [x for x in b_ex_cents if x not in f_ex_cents]
    
    
    bimg, bmsk,sr, fimg, fmsk, sub, bctd, fctd = pad_niftis(bimg, bmsk,sr, fimg, fmsk, sub, bctd, fctd)
    
    to_ax = ('I', 'A', 'L')
    bimg = reorient_to(bimg, axcodes_to=to_ax, msk=False)
    bmsk = reorient_to(bmsk, axcodes_to=to_ax)
    sr = reorient_to(sr, axcodes_to=to_ax)
    sub = reorient_to(sub, axcodes_to=to_ax, msk=False)
    bctd = reorient_centroids_to(bctd, bimg)

    
    
    
    fimg = reorient_to(fimg, axcodes_to=to_ax, msk=False)
    fmsk = reorient_to(fmsk, axcodes_to=to_ax)
    fctd = reorient_centroids_to(fctd, fimg)
    
    zms = (1,1,1)
    bzms = (1,1,1)
    fzms = (1,1,1)
  
    
    bimg_data = np.asanyarray(bimg.dataobj, dtype=np.int16)
    bmsk_data = np.asanyarray(bmsk.dataobj, dtype=bmsk.dataobj.dtype)
    
    fimg_data = fimg.get_fdata()
    fmsk_data = np.asanyarray(fmsk.dataobj, dtype=fmsk.dataobj.dtype)
    


    res_data = np.asanyarray(sub.dataobj, dtype=sub.dataobj.dtype)

    
        
    try:
        bsag_img, bcor_img, _, _, _ = sag_cor_curveprojection(bctd, bimg_data)
        bsag_msk, bcor_msk, y_cord, z_cord, x_ctd = sag_cor_curveprojection(bctd, bmsk_data)
        res_im, res_cor, _, _, _ = sag_cor_curveprojection(bctd, res_data)

        bsag_img, res_im, new_bctd = create_interpolated_im(bsag_img,res_im,x_ctd,z_cord,30,bctd)
        _, bsag_msk, _ = create_interpolated_im(bsag_img,bsag_msk,x_ctd,z_cord,30,bctd)
        bcor_img, res_cor, new_bcent_cor = create_interpolated_im(bcor_img,res_cor,x_ctd,y_cord,30,bctd)
        _, bcor_msk, _ = create_interpolated_im(bcor_img,bcor_msk,x_ctd,y_cord,30,bctd)

        
    except Exception as e:
        if len(bctd)==2:
            l = 1
        else:
            l = int(len(bctd)/2)
        bsag_img = bimg_data[:,:,int(bctd[l][3])]
        bsag_msk = bmsk_data[:,:,int(bctd[l][3])]
        bcor_img = bimg_data[:,int(bctd[l][2]),:]
        bcor_msk = bmsk_data[:,int(bctd[l][2]),:]
        res_cor = res_data[:,int(bctd[l][2]),:]
        new_bctd = bctd
        new_bcent_cor = bctd
        res_im = res_data[:,int(bctd[l][2]),:]
        
       
    try:
        fsag_img, fcor_img,y_cord, z_cord, x_ctd = sag_cor_curveprojection(fctd, fimg_data)
        fsag_img, _, new_fctd = create_interpolated_im(fsag_img,fsag_img,x_ctd,z_cord,30,fctd)
        fcor_img, _, new_fcent_cor = create_interpolated_im(fcor_img,fcor_img,x_ctd,y_cord,30,fctd)
    except Exception as e:
        if len(fctd)==2:
            l = 1
        else:
            l = int(len(fctd)/2)
        fsag_img = fimg_data[:,:,int(fctd[l][3])]
        fcor_img = fimg_data[:,int(fctd[l][2]),:]
        new_fctd = fctd
        new_fcent_cor = fctd
    
    bdrr_iso = deepcopy(bimg_data.astype(float))
    bdrr_iso[bmsk_data==0] = np.nan
    drr_b = np.nansum(bdrr_iso,2)
        

    fdrr_iso = deepcopy(fimg_data.astype(float))
    fdrr_iso[fmsk_data==0] = np.nan
    drr_f = np.nansum(fdrr_iso,2)
    
    for x in to_remove_bs:
        bsag_msk[bsag_msk==x] = 0
        bcor_msk[bcor_msk==x] = 0


    fig, axs, size = create_figure_1(dpi,drr_b,None)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    axs.imshow(drr_b, cmap=plt.cm.gray, interpolation='lanczos')
    

    plot_sag_centroids(axs, bctd, zms)
    plot_bmd_label(axs,bctd,zms, bbmd)
    buffer_ = BytesIO()
    plt.savefig(buffer_, format = "png")
    buffer_.seek(0)
    image = PIL.Image.open(buffer_)
    b_sag_drr = np.asarray(image)
    buffer_.close()
    plt.close()
    
    

    
    fig, axs,_ = create_figure_1(dpi,drr_f,size)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    axs.imshow(drr_f, cmap=plt.cm.gray, interpolation='lanczos')
    plot_sag_centroids(axs, fctd, zms)
    plot_bmd_label(axs,fctd,zms, fbmd)
    buffer_ = BytesIO()
    plt.savefig(buffer_, format = "png")
    buffer_.seek(0)
    image = PIL.Image.open(buffer_)
    f_sag_drr = np.asarray(image)
    buffer_.close()
    plt.close()
    

    bsag_msk[bsag_msk>0] = 2
    res_im[res_im>0] = 1
    
    bsag_msk
    bsag_msk[res_im>0]  = np.nan
    
    bsag_msk[bsag_msk==0] = np.nan
    res_im[res_im==0] = np.nan
    bsag_msk[0,0] = 1
    res_im[0,0] = 2
    

    
        
    fig, axs, size = create_figure_1(dpi,bsag_img,size)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    axs.imshow(bsag_img, cmap=plt.cm.gray, norm=wdw_sbone)
    axs.imshow(bsag_msk, cmap=cm_sr, alpha=0.7)
    axs.imshow(res_im, cmap=cm_sr, alpha=0.2)
    
    
    plot_sag_centroids(axs, new_bctd, zms)
    if not b_labels.empty: 
        plot_sag_labels(axs, new_bctd, zms, b_labels, size=2, text=True)
    
    buffer_ = BytesIO()
    plt.savefig(buffer_, format = "png")
    buffer_.seek(0)
    image = PIL.Image.open(buffer_)
    bsag_img_ = np.asarray(image)
    buffer_.close()
    plt.close()
    
    fig, axs,fsize = create_figure_1(dpi,fsag_img,size)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    axs.imshow(fsag_img, cmap=plt.cm.gray, norm=wdw_sbone)
    plot_sag_centroids(axs, new_fctd, fzms)
    if not f_labels.empty: 
        plot_sag_labels(axs, new_fctd, zms, f_labels, size=2, text=True)
    buffer_ = BytesIO()
    plt.savefig(buffer_, format = "png")
    buffer_.seek(0)
    image = PIL.Image.open(buffer_)
    fsag_img_ = np.asarray(image)

    buffer_.close()
    plt.close()
    ###################################
    
    bcor_msk[bcor_msk>0] = 2
    res_cor[res_cor>0] = 1
    
    
    bcor_msk[res_cor>0]  = np.nan
    
    bcor_msk[bcor_msk==0] = np.nan
    res_cor[res_cor==0] = np.nan
    bcor_msk[0,0] = 1
    res_cor[0,0] = 2
    
    
    fig, axs,fsize = create_figure_1(dpi,bcor_img,size)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    axs.imshow(bcor_img, cmap=plt.cm.gray, norm=wdw_sbone)
    axs.imshow(bcor_msk, cmap=cm_sr, alpha=0.7)
    axs.imshow(res_cor, cmap=cm_sr, alpha=0.2)
    plot_cor_centroids(axs, new_bcent_cor, fzms)

    buffer_ = BytesIO()
    plt.savefig(buffer_, format = "png")
    buffer_.seek(0)
    image = PIL.Image.open(buffer_)
    bcor_img_ = np.asarray(image)

    buffer_.close()
    plt.close()
    
    
    fig, axs,fsize = create_figure_1(dpi,fcor_img,size)
    fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    axs.imshow(fcor_img, cmap=plt.cm.gray, norm=wdw_sbone)
    plot_cor_centroids(axs, new_fcent_cor, fzms)

    buffer_ = BytesIO()
    plt.savefig(buffer_, format = "png")
    buffer_.seek(0)
    image = PIL.Image.open(buffer_)
    fcor_img_ = np.asarray(image)

    buffer_.close()
    plt.close()
    
    
    return bimg, b_sag_drr, f_sag_drr, bctd, fimg, bsag_img_,fsag_img_, fctd, bcor_img_, fcor_img_,bzms,fzms, size, fsize

def plot_sag_labels(axs, ctd, zms, ctd_labels, size=2, text=True):
    # requires v_dict = dictionary of mask labels and a pd dataframe with labels "ctd_labels"
    for v in ctd[1:]:
        if v[0] < 8 or v[0] >28 or v[0]==26:
            continue
        if ctd_labels['def_'+v_dict[v[0]]].values > 0:            # handle non-existing labels
            vert_label = int(ctd_labels['def_'+v_dict[v[0]]])
        else:
            vert_label = 0 
        if vert_label > 0:            
            axs.add_patch(Circle((v[2]*zms[1], v[1]*zms[0]), size, color=colors_labels[vert_label]))
            if text:
                axs.text(v[2]*zms[1]- 11, v[1]*zms[0]+2, vert_label, fontdict={'color': cm_labl(vert_label), 'weight': 'bold'})

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
            if d['label'] ==26:
                continue
            ctd_list.append([d['label'], d['X'], d['Y'], d['Z']]) 
    return ctd_list

def create_interpolated_im(img,msk,x_ctd,y_cord,inter,ctd):
    #interpolates a given 2D image 'img' along with its mask 'msk' to an interpolation interval 'inter' using the distance dictionart l_dict  
    
    #create a dictionary of euclidean distances between each slice
    xcords = range(x_ctd[0],x_ctd[-1],1)
    distances = {}
    co = 0
    cx = 0
    cy = 0
    for x,y in zip(xcords,y_cord):

        if co>0:
            d =math.sqrt((x -cx)**2 + (y -cy)**2)
            distances[x] = d
        else:
            distances[x] = 1
        co+=1
        cx = x
        cy = y
    interp_ctd = []

    ks = list(sorted(distances.keys()))
    #split the spine into chunks depending on the value of inter
    chunks = [ks[x:x+inter] for x in range(0, len(ks), inter)]

    n_d = {}
    n_d1 = {}
    for iii,ch in enumerate(chunks):
        
        n_d[ch[0]+inter] = sum([distances[y] for y in ch])
    #calculate new centroids

    new_cent = deepcopy(ctd)

    for cent in new_cent:
        if type(cent) is list:
            su =sum([distances[i]-1 for i in distances if i<cent[1]])
            cent[1]+= su
    l_dict = n_d

    x = l_dict.values()
    h,w = img.shape
    
    first_o=0
    ixx_img =None
    ixx_msk = None
    for k in sorted(l_dict.keys()):
        new_slices = 0    
        val = l_dict[k]

        if val >= inter:
            new_slices = int(round(val-inter))
        if new_slices > 0:
            hh = k-first_o+new_slices

            nimg = resize(img[first_o:k,:],(hh,w),order = 3,preserve_range=True, mode='constant',anti_aliasing=False)
            nmsk = resize(msk[first_o:k,:],(hh,w),order = 0,preserve_range=True, mode='constant',anti_aliasing=False)
                        
            if ixx_img is None:
                ixx_img = nimg     
                ixx_msk= nmsk
            else:
                ixx_img = np.concatenate((ixx_img,nimg),axis = 0)
                ixx_msk = np.concatenate((ixx_msk,nmsk),axis = 0)
            first_o = k
        else:
            if ixx_img is None:
                ixx_img = img[first_o:k,:]  
                ixx_msk= msk[first_o:k,:]
            else:    
                ixx_img = np.concatenate((ixx_img,img[first_o:k,:]),axis = 0)
                ixx_msk = np.concatenate((ixx_msk,msk[first_o:k,:]),axis = 0)
            first_o = k
    ixx_img = np.concatenate((ixx_img,img[k:,:]),axis=0)
    ixx_msk=np.concatenate((ixx_msk,msk[k:,:]),axis=0)
    return ixx_img,ixx_msk,new_cent



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
    fig.patch.set_facecolor('black')
    ax_neu = fig.add_subplot(1,1,1) 
    plt.axis('off')

    ax_neu.imshow(cor)
    fig.canvas.draw()
    
def cor_f(slc,img,zms,size):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img_median = img.get_fdata()[:,slc,:]
        
        
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

    ax_neu.imshow(img_median, cmap="gray", norm=wdw_sbone)
    fig.canvas.draw()
#     display(fig)


# v_dict = {
#     1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7',
#     8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6', 14: 'T7',
#     15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12', 20: 'L1',
#     21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'L6', 28: 'T13'
# }
def get_raw_ct_paths(directory):
    im_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("_ct.nii.gz"):
                im_paths.append(os.path.join(root, file))
    return im_paths

def get_derivative_path_from_raw_ct(img_path):
    drv = str(img_path).replace("rawdata", "derivatives")
    
    msk_path = drv.replace('_ct.nii.gz', '_seg-vert_msk.nii.gz')
    sr_path = drv.replace('_ct.nii.gz', '_seg-subreg_msk.nii.gz')
    ctd_path = sr_path.replace('_msk.nii.gz', '_ctd.json')
    return msk_path, sr_path, ctd_path

def get_baseline_fu_ses(unique_sessions):
    ses1 = int(unique_sessions[0][4:])
    ses2 = int(unique_sessions[1][4:])

    if ses1 > ses2:
        baseline_ses = unique_sessions[1]
        followup_ses = unique_sessions[0]
    else:
        baseline_ses = unique_sessions[0]
        followup_ses = unique_sessions[1]
    return baseline_ses, followup_ses  

def get_study_dict(fold, baseline_ses, followup_ses):

    baseline_dir = os.path.join(fold, baseline_ses)
    followup_dir = os.path.join(fold, followup_ses)

    baseline_ims = [os.path.join(baseline_dir, x) for x in os.listdir(baseline_dir) if x.endswith('_ct.nii.gz')]
    followup_ims = [os.path.join(followup_dir, x)  for x in os.listdir(followup_dir) if x.endswith('_ct.nii.gz')]

    study_dict = {'baselines': baseline_ims, 'followups': followup_ims}
    return study_dict


def get_sub_nifti_path(baseline_path, followup_path):
    sub_dir = os.path.join(os.path.dirname(os.path.dirname(baseline_path)).replace('rawdata','derivatives'), 'subtraction')
    baseline_ses = os.path.basename(os.path.dirname(baseline_path))
    followup_ses = os.path.basename(os.path.dirname(followup_path))
    spl_b, spl_f = '', ''
    if 'split' in baseline_path:
        spl_b = '_split-{}'.format(baseline_path.split('split-')[1].split('_')[0])
    if 'split' in followup_path:
        spl_f = '_split-{}'.format(followup_path.split('split-')[1].split('_')[0])

    sub_name = os.path.basename(baseline_path).split('_')[0] + '_sesb-{}{}'.format(baseline_ses.split('-')[1],spl_b) + '_sesf-{}{}'.format(followup_ses.split('-')[1],spl_f) + '_msk.nii.gz'
    sub_path = os.path.join(sub_dir, sub_name)
    
    return sub_path

def create_evaluation_path(results_folder, sub_path, rater):
    return os.path.join(results_folder,os.path.basename(sub_path).replace('_ref.nii.gz', '_rater-{}_eval.json'.format(rater)))

def get_paths_followup(dataset_folder, ex, rater,chunks=None, results_folder='./results'):
    main_fold = os.path.join(dataset_folder, 'rawdata')
    folders = [os.path.join(main_fold,x) for x in os.listdir(main_fold) if 'sub-ctfu' in x]
    folders.sort()
    if chunks is not None:
        pointer1 = chunks * 20
        pointer2 = pointer1 + 20
    else:
        pointer1 = 0
        pointer2 = len(folders)
    paths = []
    for fold in folders[pointer1:pointer2]:
        unique_sessions = [x for x in os.listdir(fold) if 'ses' in x]
        if len(unique_sessions) != 2:
            continue
        baseline_ses, followup_ses = get_baseline_fu_ses(unique_sessions)
        print('dataset_folder ', fold)
        print('baseline: ', baseline_ses)
        print('fu: ', followup_ses)
        
        
        study_dict = get_study_dict(fold, baseline_ses, followup_ses)
        print(study_dict)
        
        for baseline_path in study_dict['baselines']:
            for followup_path in study_dict['followups']:
                sub_path = get_sub_nifti_path(baseline_path, followup_path)
                print('subpath ', sub_path)
                eval_path = create_evaluation_path(results_folder, sub_path, rater)
                if ex:
                    if os.path.isfile(eval_path):
                        print('continued')
                        continue
                        
                
                b_msk_path, b_sr_path, b_ctd_path = get_derivative_path_from_raw_ct(baseline_path)
                f_msk_path, f_sr_path, f_ctd_path = get_derivative_path_from_raw_ct(followup_path)
                
                
                if os.path.isfile(sub_path):
                    print(baseline_path)
                    ps = [baseline_path, b_msk_path,b_sr_path,b_ctd_path,followup_path,f_msk_path,f_sr_path,f_ctd_path,sub_path]
                    paths.append(ps)
                else:
                    print('stoopid')
            print('#############')
    return paths
                
def load_json(json_path):
    name = Path(json_path).name
    name = str(name).split('_')[0]
    with open(json_path) as json_data:
        js = json.load(json_data)
        js['name'] = name
        json_data.close()
    return js

def get_results_fu(path,vert):
    try:
        entry = get_csv_entry(path)
        labels = pd.read_csv('./labels.csv')

        label  = labels[vert][labels['ID']==entry]

        if label.values > 0:            # handle non-existing labels
            vert_label = str(int(label))
        else:
            vert_label = "" 
    except Exception as e:
        print('error ', str(e))
        vert_label = ""
    return vert_label

def get_csv_entry(img_path):
    csv_name = str(os.path.basename(img_path).replace('_ct.nii.gz', ''))
    return csv_name


def get_paths_from_case(case_path):
    cases = []
    unique_sessions = [x for x in os.listdir(case_path) if 'ses' in x]
    baseline_ses, followup_ses = get_baseline_fu_ses(unique_sessions)
    study_dict = get_study_dict(case_path, baseline_ses, followup_ses)

    for baseline_path in study_dict['baselines']:
        for followup_path in study_dict['followups']:
            sub_path = get_sub_nifti_path(baseline_path, followup_path)

            if os.path.isfile(sub_path):
                p_dict = {
                    'b_img_pth': baseline_path,
                    'f_img_pth': followup_path,
                    'sub_pth'  : sub_path,
                }
                cases.append(p_dict)    
    return cases

def get_rater_cases_from_csv(data_frame, dataset_path,rater=None, load_all=False, exclude_rated=False):
    labelled_IDs = pd.read_csv('./CTFU_Fx_{}.csv'.format(str(rater)))['ID'].tolist()
    
    rater_cases = []
    for idx, row in data_frame.iterrows():

        if load_all:
            rater_cases.append(row['case_name'])
            
        else:
            if row['rater'] == rater or row['multirater'] ==1:
                rater_cases.append(row['case_name'])
    rater_cases= list(set(rater_cases))
    rater_cases.sort()
    cases_paths = []
    for case in rater_cases:
        cs_pth = os.path.join(os.path.join(dataset_path,'rawdata'), case.lower())
        if not os.path.isdir(cs_pth):
            case = case.lower().replace('ctfu', 'ctfu0')
            cs_pth = os.path.join(os.path.join(dataset_path,'rawdata'), case.lower())
            if not os.path.isdir(cs_pth):
                print(cs_pth)
                print('cpth no ex ',case)
                continue
        try:
            case_p = get_paths_from_case(os.path.join(os.path.join(dataset_path,'rawdata'), case))
        except Exception:
            print('No Followup Found ',case)
            continue
        if not case_p:
            print('other no fu ',case)
            continue
        if exclude_rated:
            for idx,c in enumerate(case_p):
                ID_b = os.path.basename(c['b_img_pth']).replace('_ct.nii.gz', '')
                ID_f = os.path.basename(c['f_img_pth']).replace('_ct.nii.gz', '')
                if ID_b in labelled_IDs and ID_f in labelled_IDs:
                    case_p.pop(idx)
        cases_paths = cases_paths + case_p
    return cases_paths