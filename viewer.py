from ipywidgets import GridspecLayout, Button, Layout, ButtonStyle, GridBox, interact, interactive, fixed, Layout
import ipywidgets as widgets
import numpy as np
from utils import *
from IPython.display import display, clear_output
from pathlib import Path
import json
import pandas as pd

class viewer:
    def __init__(self, rater_dict, save_pth, labels, exclude_c=False, display_coronal=True):
        
        img, sub1, sub2, ctd, fimg, sag1,sag2, fctd, bcor_img_, fcor_img_, zms, fzms, size, fsize = process_data_snp_sub(rater_dict, labels, dpi=120)
        
        b_img_path = rater_dict['b_img_pth']
        f_img_path = rater_dict['f_img_pth']

        self.database = pd.read_csv(save_pth, dtype=str)
        self.b_img_path = b_img_path
        self.f_img_path = f_img_path
        self.img = img
        self.fimg = fimg
        self.sag1 = sag1
        self.sag2 = sag2
        self.bcor = bcor_img_
        self.fcor = fcor_img_
        self.zms = zms
        self.fzms = fzms
        self.sub1 = sub1
        self.sub2 = sub2
        self.ctd = ctd
        self.fctd = fctd
        self.save_pth = save_pth
        self.values = ['']*(len(ctd)-1)
        self.verts = []
        self.size =size
        self.fsize =fsize
        self.display_coronal = display_coronal
        self.bname = os.path.basename(str(b_img_path)).replace('_ct.nii.gz','')
        self.fname = os.path.basename(str(f_img_path)).replace('_ct.nii.gz','')
        self.colors = np.array([
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
        ctd_dict1 = {x[0] : [x[1], x[2], x[3]] for x in ctd[1:]}
        ctd_dict2 = {x[0] : [x[1], x[2], x[3]] for x in fctd[1:]}
        self.full_ctd =  list(set(list(ctd_dict1.keys())+list(ctd_dict2.keys())))
        self.display_ctd = self.full_ctd.copy()
        if exclude_c:
            self.display_ctd = [x for x in self.full_ctd if x>7]
        self.v_dict = {
        1:'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7',
        8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6', 14: 'T7',
        15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12', 20: 'L1',
        21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'L6', 28: 'T13'
    }
        id_bs = str(os.path.basename(b_img_path).replace('_ct.nii.gz', ''))
        id_fu = str(os.path.basename(f_img_path).replace('_ct.nii.gz', ''))
        if id_bs in self.database.ID.values:
            self.bs_presence = "## Baseline already graded - displaying your previous grades - press save to overwrite\n"
        else:
            self.bs_presence = ""
        
        if id_fu in self.database.ID.values:
            self.fu_presence = "## Followup already graded - displaying your previous grades - press save to overwrite"
        else:
            self.fu_presence = ""
        self.initial_disp_txt = self.bs_presence + self.fu_presence
        
        con_full_list = [self.v_dict[i] for i in self.full_ctd]
        if exclude_c:
            con_full_list = [self.v_dict[i] for i in self.full_ctd if i>7]
        if id_bs in self.database.ID.values:
            
            dictt = self.database[self.database['ID']==id_bs].replace({np.nan:''}).to_dict(orient='records')
            self.bs_gr = {k: (str(v) if k in con_full_list else '') for k,v in dictt[0].items() if k in  con_full_list }
        else:
            l_dict_b = get_labels_from_imp(labels, b_img_path)
            self.bs_gr = {i:(str(int(l_dict_b['def_'+i])) if 'def_'+i in l_dict_b else '') for i in con_full_list}
            
#             self.bs_gr = {self.v_dict[i]:(l_dict_b['def_'+self.v_dict[i]] if 'def_'+self.v_dict[i] in l_dict_b else '') for i in con_full_list}

        if id_fu in self.database.ID.values:
            dictt = self.database[self.database['ID']==id_fu].replace({np.nan:''}).to_dict(orient='records')

            self.fu_gr =  {k: (str(v) if k in con_full_list else '') for k,v in dictt[0].items() if k in  con_full_list}
        else:
            l_dict_f = get_labels_from_imp(labels, f_img_path)
            self.fu_gr = {i:(str(int(l_dict_f['def_'+i])) if 'def_'+i in l_dict_f else '') for i in con_full_list}




                         
        

    def create_fig(self):
        
        def on_NEXT(b):
            fu_values = [i.value for i in  self.values.children[2].children]
            bs_values= [i.value for i in  self.values.children[1].children]
            
            keys =[i for i in  self.full_ctd]
            bs_dict= {self.v_dict[x]:str(y) for x,y in zip(keys,bs_values)}
            saved1 = populate_database(self.save_pth, bs_dict, self.b_img_path,'baseline')
            fu_dict = {self.v_dict[x]:str(y) for x,y in zip(keys,fu_values)}
            saved2 = populate_database(self.save_pth, fu_dict, self.f_img_path,'followup')
            self.header1.value= '### Evaluation Saved!!'

            
        def rgb2hex(col):
            return "#{:02x}{:02x}{:02x}".format(col[0],col[1],col[2])
        
        boxes = [widgets.Text( value=self.bs_gr[self.v_dict[i]],layout= Layout(width='45px', height='20px')) for i in self.display_ctd if type(i) is not tuple ]
        boxes1 =[widgets.Text( value=self.fu_gr[self.v_dict[i]],layout= Layout(width='45px', height='20px')) for i in self.display_ctd if type(i) is not tuple ]
#         keys = [widgets.Text( value=v_dict[i[0]],layout= Layout(width='50px', height='30px') )for i in self.ctd if type(i) is not tuple ]
        self.verts = [self.v_dict[i[0]] for i in self.ctd if type(i) is not tuple]
        keys = [widgets.Label( value=r'\(\color{'+str(rgb2hex(self.colors[i-1]))+ '}{' + self.v_dict[i]  + '}\)',layout= Layout(width='45px', height='20px') )for i in self.display_ctd if type(i) is not tuple ]
         
        left_box = widgets.VBox(boxes, description='FXG')
        
        right_box = widgets.VBox(keys, description='vert')
        ivd_box = widgets.VBox(boxes1, description='IVD')
        ph=widgets.HBox([right_box,left_box ,ivd_box], layout=Layout(top='40px', height='100%'))  

        width = '5'
        # x = save_dict()
        if self.display_coronal:
            views = 9
            c1,c2,c3,c4, c5 = 4,5,6,7,8
        else:
            views = 7
            c1,c2,c3,c4, c5 = 0,0,4,5,6
        grid = GridspecLayout(1, views, height='auto')
        grid[0, 0] = widgets.interactive(sag_f, slc=widgets.IntSlider(min=0, max=self.img.shape[2]-1, step =1, value=int(self.img.shape[2]/2),layout=Layout(width='120%',left='-25px')),img=fixed(self.img),zms=fixed(self.zms),layout= Layout(width='{}%'.format(width),height='100%'))
        grid[0, 1] = widgets.interactive(sag_f, slc=widgets.IntSlider(min=0, max=self.fimg.shape[2]-1, step =1, value=int(self.fimg.shape[2]/2),layout=Layout(width='120%',left='-25px')),img=fixed(self.fimg),zms=fixed(self.fzms),size=fixed(self.size),layout= Layout(width='{}%'.format(width),height='100%'))
        grid[0, 2] = widgets.interactive(sag_i, x=widgets.IntSlider(min=0, max=1, step =1, value=0, layout=Layout(visibility='hidden',width='100%')), sag=fixed(self.sag1),layout= Layout(width='{}%'.format(width),height='100%'))
        grid[0, 3] = widgets.interactive(sag_i, x=widgets.IntSlider(min=0, max=1, step =1, value=0, layout=Layout(visibility='hidden',width='100%')) , sag=fixed(self.sag2) ,layout= Layout(width='{}%'.format(width),height='100%'))
        if self.display_coronal:
            grid[0, c1] = widgets.interactive(cor_i, x=widgets.IntSlider(min=0, max=1, step =1, value=0, layout=Layout(visibility='hidden',width='100%')), cor=fixed(self.bcor),layout= Layout(width='{}%'.format(width),height='100%'))
            grid[0, c2] = widgets.interactive(cor_i, x=widgets.IntSlider(min=0, max=1, step =1, value=0, layout=Layout(visibility='hidden',width='100%')), cor=fixed(self.fcor),layout= Layout(width='{}%'.format(width),height='100%'))
        
        grid[0, c3] = widgets.interactive(sag_drr, x=widgets.IntSlider(min=0, max=1, step =1, value=0, layout=Layout(visibility='hidden',width='100%')), drr=fixed(self.sub1),layout= Layout(width='{}%'.format(width),height='100%'))
        grid[0, c4] = widgets.interactive(cor_i, x=widgets.IntSlider(min=0, max=1, step =1, value=0, layout=Layout(visibility='hidden',width='100%')), cor=fixed(self.sub2),layout= Layout(width='{}%'.format(width),height='100%'))
        grid[0, c5] = ph
        self.values = grid[0, c5]

        head = Button(description='{} - {}'.format(self.bname,self.fname),
                         layout=Layout(width='auto', grid_area='header'),
                         style=ButtonStyle(button_color='black'))
        
        header  = Button(description='Save and Continue',
                         layout=Layout(width='auto', grid_area='footer'),
                         style=ButtonStyle(button_color='gray'))
#         header1  = Button(description='di fi',
#                          layout=Layout(width='auto', grid_area='display'),
#                          style=ButtonStyle(button_color='gray'))
        
        self.header1  = widgets.Textarea(value=self.initial_disp_txt,
                         layout=Layout(width='auto', grid_area='display'))
        
        header.on_click(on_NEXT)
        grd = GridBox(children=[head, grid, header, self.header1],
                layout=Layout(
                    width='auto',
                    grid_template_rows='auto auto',
                    grid_template_columns='100%',
                    grid_template_areas='''
                    "header header header header"
                    "main main . sidebar "
                    "footer footer footer footer"
                    "display display display display"
                    ''')
               )
        return grd