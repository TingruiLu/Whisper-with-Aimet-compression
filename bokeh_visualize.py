from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource
from bokeh.layouts import column
from decimal import Decimal
import pickle

# Load the data
comp_ratios_path = "./data/greedy_selection_comp_ratios_list.pkl"
eval_scores_path = "./data/greedy_selection_eval_scores_dict.pkl"

with open(comp_ratios_path, 'rb') as f1:
    data1 = pickle.load(f1)

with open(eval_scores_path, 'rb') as f2:
    data2 = pickle.load(f2)

data = data2

# define layers
layer_types = ['q_proj', 'k_proj', 'v_proj', 'out_proj']
plots = []  

for layer_type in layer_types:

    ratios = sorted({float(ratio) for scores in data.values() for ratio in scores.keys()})
    layer_scores_by_ratio = {ratio: [] for ratio in ratios}

    # group the data
    for layer, scores in data.items():
        if layer_type in layer:
            for ratio, value in scores.items():
                layer_scores_by_ratio[float(ratio)].append(value)

    # create a data source
    sources = {}
    for ratio, scores in layer_scores_by_ratio.items():
        sources[ratio] = ColumnDataSource(data={
            'x': list(range(len(scores))),  
            'y': scores                     
        })

    # create bokeh graph
    p = figure(title=f"{layer_type.capitalize()} Scores Across Layers",
               x_axis_label="Layer Index",
               y_axis_label="Score",
               width=800, height=400)

    # plot
    for ratio, source in sources.items():
        p.line('x', 'y', source=source, line_width=2)

    
    plots.append(p)

layout = column(*plots)


curdoc().add_root(layout)
