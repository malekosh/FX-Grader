from ipywidgets import GridspecLayout, Button, Layout, ButtonStyle, GridBox, interact, interactive, fixed, Layout
import ipywidgets as widgets
import numpy as np
from utils import *
from IPython.display import display, clear_output
from pathlib import Path
import json

class viewer:
    def __init__(self, img, sag, cor, zms, drr, ctd, save_pth,size,zeros,load_rater):
        self.img = img
        self.sag = sag
        self.cor = cor
        self.zms = zms
        self.drr = drr
        self.ctd = ctd
        self.save_pth = save_pth
        self.values = ['']*(len(ctd)-1)
        self.verts = []
        self.size =size
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
        self.v_dict = {
        1:'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7',
        8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6', 14: 'T7',
        15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12', 20: 'L1',
        21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'L6', 28: 'T13'
    }
        if load_rater:
            self.ivd_gr  = {self.v_dict[i[0]]:'0' if v_dict[i[0]] in ['L1', 'L2', 'L3', 'L4', 'L5', 'L6'] else ''  for i in self.ctd if type(i) is not tuple}
            if load_rater.lower() == 'jan':
                self.fx_gr = {self.v_dict[i[0]]:get_result_jan(save_pth,self.v_dict[i[0]]) for i in self.ctd if type(i) is not tuple}
            elif load_rater.lower() == 'max':
                self.fx_gr = {self.v_dict[i[0]]:get_result_max(save_pth,self.v_dict[i[0]]) for i in self.ctd if type(i) is not tuple}
            elif load_rater.lower() == 'thomas':
                self.fx_gr = {self.v_dict[i[0]]:get_result_thomas(save_pth,self.v_dict[i[0]]) for i in self.ctd if type(i) is not tuple}
            else:
                self.fx_gr = {self.v_dict[i[0]]:'0' for i in self.ctd if type(i) is not tuple}
            
        elif zeros:
            if Path(self.save_pth).is_file():
                with open(self.save_pth) as json_data:
                    self.fx_gr = json.load(json_data)
                    json_data.close()
            else:
                self.fx_gr = {self.v_dict[i[0]]:'0' for i in self.ctd if type(i) is not tuple}
            
            if Path(self.save_pth.replace('_fx-', '_ivd-')).is_file():
                with open(self.save_pth.replace('_fx-', '_ivd-')) as json_data:
                    self.ivd_gr = json.load(json_data)
                    json_data.close()
            else:
                self.ivd_gr  = {self.v_dict[i[0]]:'0' if v_dict[i[0]] in ['L1', 'L2', 'L3', 'L4', 'L5', 'L6'] else ''  for i in self.ctd if type(i) is not tuple}
        else:
            self.fx_gr = {self.v_dict[i[0]]:'' for i in self.ctd if type(i) is not tuple}
            self.ivd_gr = {self.v_dict[i[0]]:'' for i in self.ctd if type(i) is not tuple}

    def create_fig(self):
        
        def on_NEXT(b):
            ivd_values = [i.value for i in  self.values.children[2].children]
            values= [i.value for i in  self.values.children[1].children]
            keys =[i for i in  self.verts]
            new_dict= {x:y for x,y in zip(keys,values)}
            ivd_dict = {x:y for x,y in zip(keys,ivd_values)}
            save_json(new_dict, self.save_pth)
            save_json(ivd_dict, self.save_pth.replace('_fx-', '_ivd-'))
            clear_output()
            
        def rgb2hex(col):
            return "#{:02x}{:02x}{:02x}".format(col[0],col[1],col[2])
        
        boxes = [widgets.Text( value=self.fx_gr[self.v_dict[i[0]]],layout= Layout(width='60px', height='30px')) for i in self.ctd if type(i) is not tuple ]
        boxes1 =[widgets.Text( value=self.ivd_gr[self.v_dict[i[0]]],layout= Layout(width='60px', height='30px')) for i in self.ctd if type(i) is not tuple ]
#         keys = [widgets.Text( value=v_dict[i[0]],layout= Layout(width='50px', height='30px') )for i in self.ctd if type(i) is not tuple ]
        self.verts = [self.v_dict[i[0]] for i in self.ctd if type(i) is not tuple]
        keys = [widgets.Label( value=r'\(\color{'+str(rgb2hex(self.colors[i[0]-1]))+ '}{' + self.v_dict[i[0]]  + '}\)',layout= Layout(width='50px', height='30px') )for i in self.ctd if type(i) is not tuple ]
         
        left_box = widgets.VBox(boxes, description='FXG')
        
        right_box = widgets.VBox(keys, description='vert')
        ivd_box = widgets.VBox(boxes1, description='IVD')
        ph=widgets.HBox([right_box,left_box,ivd_box ], layout=Layout(top='40px', height='100%'))  


        # x = save_dict()
        grid = GridspecLayout(1, 6, height='auto')
        grid[0, 0] = widgets.interactive(sag_f, slc=widgets.IntSlider(min=0, max=self.img.shape[2]-1, step =1, value=int(self.img.shape[2]/2),layout=Layout(width='120%',left='-25px')),img=fixed(self.img),zms=fixed(self.zms),layout= Layout(width='12%',height='100%'))
        grid[0, 1] = widgets.interactive(cor_f, slc=widgets.IntSlider(min=0, max=self.img.shape[1]-1, step =1, value=int(self.img.shape[1]/2),layout=Layout(width='120%',left='-25px')),img=fixed(self.img),zms=fixed(self.zms),size=fixed(self.size),layout= Layout(width='12%',height='100%'))
        grid[0, 2] = widgets.interactive(sag_drr, x=widgets.IntSlider(min=0, max=1, step =1, value=0, layout=Layout(visibility='hidden',width='100%')), drr=fixed(self.drr),layout= Layout(width='12%',height='100%'))
        grid[0, 3] = widgets.interactive(sag_i, x=widgets.IntSlider(min=0, max=1, step =1, value=0, layout=Layout(visibility='hidden',width='100%')) , sag=fixed(self.sag) ,layout= Layout(width='12%',height='100%'))
        grid[0, 4] = widgets.interactive(cor_i, x=widgets.IntSlider(min=0, max=1, step =1, value=0, layout=Layout(visibility='hidden',width='100%')), cor=fixed(self.cor),layout= Layout(width='12%',height='100%'))
        grid[0, 5] = ph
        self.values = grid[0, 5]


        header  = Button(description='Save and Continue',
                         layout=Layout(width='auto', grid_area='footer'),
                         style=ButtonStyle(button_color='red'))
        header.on_click(on_NEXT)
        grd = GridBox(children=[grid, header],
                layout=Layout(
                    width='100%',
                    grid_template_rows='auto auto',
                    grid_template_columns='100%',
                    grid_template_areas='''
                    "header header header header"
                    "main main . sidebar "
                    "footer footer footer footer"
                    ''')
               )
        return grd