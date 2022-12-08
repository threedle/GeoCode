import json
import numpy as np


def gen_and_save_barplot(barplot_json_path, title, barplot_target_image_path=None):
    from matplotlib import pyplot as plt
    with open(barplot_json_path, 'r') as barplot_json_file:
        data = json.load(barplot_json_file)

    inputs_to_eval = data['inputs_to_eval']
    correct_arr_pc = data['correct_arr_pc']
    correct_arr_sketch = data['correct_arr_sketch']
    total_pc = data['total_pc']
    total_sketch = data['total_sketch']

    correct = [a + b for a, b in zip(correct_arr_pc, correct_arr_sketch)]
    accuracy_avg = [a / (total_pc + total_sketch) for a in correct]
    accuracy_pc = [a / total_pc for a in correct_arr_pc]
    accuracy_sketch = [a / total_sketch for a in correct_arr_sketch]

    overall_acc_avg = (sum(correct_arr_pc) + sum(correct_arr_sketch)) / (
                len(inputs_to_eval) * (total_pc + total_sketch))
    overall_acc_pc = sum(correct_arr_pc) / (len(inputs_to_eval) * total_pc)
    overall_acc_sketch = sum(correct_arr_sketch) / (len(inputs_to_eval) * total_sketch)

    is_only_sketches = False
    is_only_pcs = False
    if all([param_acc == 0 for param_acc in accuracy_pc]):
        # only sketches
        overall_acc_avg = overall_acc_sketch
        is_only_sketches = True
    if all([param_acc == 0 for param_acc in accuracy_sketch]):
        # only pcs
        overall_acc_avg = overall_acc_pc
        is_only_pcs = True

    # sort by average accuracy
    inputs_to_eval, accuracy_avg, accuracy_pc, accuracy_sketch = zip(
        *sorted(zip(inputs_to_eval, accuracy_avg, accuracy_pc, accuracy_sketch), key=lambda x: x[1]))

    inputs_to_eval += ("Overall",)
    accuracy_avg += (overall_acc_avg,)
    accuracy_pc += (overall_acc_pc,)
    accuracy_sketch += (overall_acc_sketch,)

    fig, ax = plt.subplots(figsize=(16, 14))
    X_axis = np.arange(len(inputs_to_eval)) * 2.6
    if not is_only_pcs and not is_only_sketches:
        pps = ax.barh(X_axis + 0.7, accuracy_avg, 0.7, color='steelblue')
        ax.barh(X_axis - 0.0, accuracy_pc, 0.7, color='lightsteelblue')
        ax.barh(X_axis - 0.7, accuracy_sketch, 0.7, color='wheat')
        ax.legend(labels=['Average', 'Point Clouds', 'Sketches'])
    elif is_only_pcs:
        pps = ax.barh(X_axis - 0.0, accuracy_pc, 0.7, color='lightsteelblue')
        ax.legend(labels=['Point Clouds'])
    elif is_only_sketches:
        pps = ax.barh(X_axis - 0.7, accuracy_sketch, 0.7, color='wheat')
        ax.legend(labels=['Sketches'])
    else:
        raise Exception("Either point cloud or sketch input should be processed")

    ax.bar_label(pps, fmt='%.2f', label_type='center', fontsize=8)
    ax.set_yticks(X_axis, inputs_to_eval)
    ax.set_title(title)

    if barplot_target_image_path:
        plt.savefig(barplot_target_image_path)
    return fig
