from future.builtins import input
import numpy as np
import pandas as pd
import glob, os
from os.path import dirname, join
from bokeh.layouts import row, column, widgetbox
from bokeh.models import BoxSelectTool, LassoSelectTool, HoverTool, TapTool, Div, widgets
from bokeh.models.ranges import DataRange1d
from bokeh.plotting import figure, curdoc, ColumnDataSource
# from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider
from skimage import io as skio
from skimage import measure, img_as_int
import vpipes
'''
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve --show virago_lens.py
at your command prompt (navigate to correct directory first).
It will then ask you to input a directory containing the raw experimental data (PGMs)
'''

# expt_dir = input("\nPlease type in the path to the folder that contains the IRIS data:\n")
expt_dir = '/Volumes/KatahdinHD/ResilioSync/DATA/IRIS/tCHIP_results/tCHIP004_EBOVmay@1E6'
expt_dir = expt_dir.strip('"')
os.chdir(expt_dir)
sample_name = vpipes.sample_namer(expt_dir)


txt_list = sorted(glob.glob('*.txt'))
pgm_list = sorted(glob.glob('*.pgm'))
pgm_set = set([".".join(file.split(".")[:3]) for file in pgm_list])
csv_list = sorted(glob.glob('*.csv'))
xml_list = sorted(glob.glob('*/*.xml'))
if not xml_list: xml_list = sorted(glob.glob('../*/*.xml'))
chip_name = pgm_list[0].split(".")[0]

pic_dir = '../virago_output/' + chip_name + '/processed_images'
vcount_dir =  '../virago_output/'+ chip_name + '/vcounts'

image_list = sorted(glob.glob('*.pgm'))
image_list, mirror = vpipes.mirror_finder(image_list)
image_set = sorted(list(set([".".join(image.split(".")[:3]) for image in image_list])))

data_select = chip_name + '.001.001'
def load_image(pic_dir, data_select):
    os.chdir(pic_dir)
    pic_list = sorted(glob.glob('*.png'))
    for pic_name in pic_list:
        if data_select in pic_name:
            data_name = pic_name
            pic = skio.imread(pic_name, as_gray = True)

    data_name = ".".join(data_name.split(".")[:-1])
    pic = pic[::-1, :]
    pic = measure.block_reduce(pic,block_size = (2,2))


    return pic, data_name

def load_data(vcount_dir, data_name):
    os.chdir(vcount_dir)
    vcount_csv_list = sorted(glob.glob('*.vcount.csv'))
    vcount = data_name +'.vcount.csv'
    data = pd.read_csv(vcount)
    x = data.x / 2
    y = data.y / 2
    z = data.z
    pc = data.pc
    cv_bg = data.cv_bg

    colors = ['red','green','blue']*len(pc)
    data_dict = dict(x = x, y = y, z = z, pc = pc, cv_bg = cv_bg, select = pc*4, colors = colors)

    return data_dict

img_final, data_name = load_image(pic_dir, data_select)
nrows, ncols = img_final.shape
# nrows = int(nrows / 2)
# ncols = int(ncols / 2)
os.chdir(expt_dir)
data_dict = load_data(vcount_dir, data_name)
print("Done!")

desc = Div(text=open(join(dirname(__file__), "description.html")).read(), width=300)

# rzeros = np.zeros(len(data_dict['pc']))
# rones = np.ones(len(data_dict['pc']))
source = ColumnDataSource(data=data_dict)

particle_data = HoverTool(tooltips = [("particle ID", "$index"),
                                      ("(x, y, z)", "(@x, @y, @z)"),
                                      ("percent contrast", "@pc"),
                                      ("background CV", "@cv_bg")
                                     ])
p = figure(plot_width = ncols, plot_height = nrows, min_border=10, min_border_left=5,
          x_range=(0,ncols), y_range=(nrows,0), x_axis_location=None, y_axis_location=None,
          tools = ['box_zoom,box_select,lasso_select,tap', particle_data, "reset"],
          title = data_name
)


p.image(image=[img_final], x=0, y=nrows, dw=ncols, dh=nrows)
p.select(BoxSelectTool).select_every_mousemove = False
p.select(LassoSelectTool).select_every_mousemove = False
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None


all_circles = p.circle(x = 'x', y = 'y', size = 'select', fill_color = 'colors',
                     fill_alpha = 0.25, line_color = 'blue', source = source
)
#************************************************************************************************#
def load_histo(data_dict):
    bin_no = 100
    hhist, hedges = np.histogram(data_dict['pc'], bins=bin_no)
    histo_dict = dict(hhist=hhist,
                      l_edges = hedges[:-1],
                      r_edges = hedges[1:],
                      )
    highlights = np.zeros(bin_no)
    histo_dict['highlights'] = highlights
    return histo_dict, hedges

histo_dict, hedges = load_histo(data_dict)
source2 = ColumnDataSource(data=histo_dict)

histo_data = HoverTool(tooltips =[('bin','$index'),
                                  ('percent contrast','@r_edges'),
                                  ('total particles', '@hhist'),
                                  ('particles selected', '@highlights')
                                  ],
                        )

ph = figure(plot_width=p.plot_width, plot_height=180,
            x_range = DataRange1d(start = 0, follow = 'end', range_padding = 0.1),
            y_range = DataRange1d(start = 0, follow = 'end', range_padding = 0.25),
            min_border=10, min_border_left=5, y_axis_location='left',
            tools = ['xbox_select,tap,reset',histo_data], toolbar_location = 'right',
            title = "Particle Contrast Histogram")


ph.xgrid.grid_line_color = None
ph.yaxis.major_label_orientation = np.pi/4
ph.background_fill_color = "#fafafa"
ph.select(type = TapTool)
ph.select(type = BoxSelectTool)

main_histo = ph.quad(bottom=0, left='l_edges', right='r_edges', top='hhist', source = source2,
                       alpha = 0.75, color="white", line_color="blue")
highlight_hist = ph.quad(bottom=0,
                         left='l_edges',
                         right='r_edges',
                         top='highlights',
                         source = source2,
                         alpha=0.5, color="cyan", line_color=None)



#************************************************************************************************#
def data_reloader(select_img_data, dir):
    os.chdir(expt_dir)
    vcount_dir =  '../virago_output/'+ chip_name + '/vcounts'
    data_dict = load_data(vcount_dir, select_img_data.value)
    return data_dict

def update_particles(attrname, old, new):
    data_dict = data_reloader(select_img_data, vcount_dir)
    source.data = data_dict
    all_circles = p.circle(x = 'x', y = 'y', size = 'select', source = source,
                           fill_color = 'cyan', fill_alpha = 0.25, line_color = 'blue')

def img_change(attrname, old, new):
    new_select = select_img_data.value
    print(new_select)
    os.chdir(expt_dir)
    pic_dir = '../virago_output/' + chip_name + '/processed_images'
    new_pic, data_name = load_image(pic_dir, new_select)
    nrows, ncols = new_pic.shape
    p.image(image=[new_pic], x=-0.5, y=0, dw=(ncols - 0.5), dh=nrows)
    p.title.text = new_select

def update_histo(attrname, old, new):
    data_dict = data_reloader(select_img_data, vcount_dir)
    histo_dict, hedges = load_histo(data_dict)
    hmax = max(histo_dict['hhist'])*1.1
    source2.data = histo_dict
    return histo_dict, hmax

def histo_highlighter(attr, old, new):
    inds = np.array(new['1d']['indices'])
    data_dict = data_reloader(select_img_data, vcount_dir)
    histo_dict, hedges = load_histo(data_dict)

    if len(inds) == 0 or len(inds) == len(data_dict['pc']):
        histo_dict['highlights'] = np.zeros(len(histo_dict['hhist']))
    else:
        histo_dict['highlights'], __ = np.histogram(data_dict['pc'][inds], bins=hedges)

    source2.data = histo_dict

select_img_data = widgets.Select(title="Select Image Data:", value = data_name, options = image_set)

select_img_data.on_change('value', img_change)
select_img_data.on_change('value', update_particles)
select_img_data.on_change('value', update_histo)
all_circles.data_source.on_change('selected', histo_highlighter)


def particle_select(attr, old, new):
    inds = np.array(new['1d']['indices'])
    data_dict = data_reloader(select_img_data, vcount_dir)
    histo_dict, hedges = load_histo(data_dict)
    print(len(inds))
    if (len(inds) == len(data_dict['pc'])) or len(inds) == 0:
        data_dict['select'] = data_dict['pc']*4
    # elif len(inds) == 0:
    #     data_dict['select'] = np.zeros(len(data_dict['pc']))
    else:
        selected_bins = []
        for val in inds:
            selected_bin = (histo_dict['l_edges'][val], histo_dict['r_edges'][val])
            selected_bins.append(selected_bin)
        for val in selected_bins:
            data_dict['select'][(data_dict['pc'] > val[0]) & (data_dict['pc'] <= val[1])] *= 4
    print(data_dict['select'])

    source.data = data_dict

main_histo.data_source.on_change('selected',particle_select)
#************************************************************************************************#

layout = row(column(desc, select_img_data), column(p, ph))
curdoc().add_root(layout)
curdoc().title = "VIRAGO LENS Viewer"
