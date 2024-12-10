import numpy as np
from _utils import get_non_empty_index, feature_units_dict

def get_color_series(color_index: list, 
                     cmap: str):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    color_rgb = []
    cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(color_index))
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=plt.get_cmap(cmap)) #mpl.colormaps[cmap]
    color_rgb = [scalarMap.to_rgba(c) for c in color_index]

    return color_rgb

def plot_shap_single_sample(single_shap_value, plot_title: str, plot_filename: str, target_property: str, target_property_unit: str, X_mean: np.ndarray, X_std: np.ndarray, Y_mean: np.ndarray, Y_std: np.ndarray, shap_feature_name: str):
    import matplotlib.pyplot as plt
    import shap
    import pickle
    
    # shap_value_argsort = np.argsort(np.abs(single_shap_value.values))
    plt.rcParams['xtick.minor.visible'] = False
    plt.rcParams['ytick.minor.visible'] = False
    plt.rcParams['ytick.right'] = False
    ax = shap.plots.waterfall(single_shap_value, max_display = 10, show = False)
    fig = ax.get_figure()
    for a_text in fig.get_axes()[0].texts:
        a_text.set_color("black")
    ax_average = fig.get_axes()[1]
    ax_average.set_xlim(ax.get_xlim())
    average_target_property_value = ax_average.get_xaxis().get_majorticklabels()[0]._x
    ax_average.set_xticks([average_target_property_value])
    ax_average.set_xticklabels([f"\nAverage {target_property}\n{average_target_property_value:.2f} ({target_property_unit})"], fontsize = 12)
    ax_average.get_xaxis().get_majorticklabels()[0].set_ha("left")

    target_property_value = ax.get_xaxis().get_majorticklabels()[0]._x
    feature_tick_list = []
    feature_text_list = []
    for i_feature, a_feature_text in enumerate(ax.get_yaxis().get_majorticklabels()):
        if "other features" in a_feature_text._text:
            feature_tick_list.append(a_feature_text._y)
            feature_text_list.append(f"{a_feature_text._text}")
            continue
        feature_text_tmp = a_feature_text._text.split("=")
        if len(feature_text_tmp) == 1: 
            continue
        feature_tick_list.append(a_feature_text._y)
        shap_feature_name_index = shap_feature_name.index(feature_text_tmp[1].strip())

        feature_name_unit = feature_units_dict.get(shap_feature_name[shap_feature_name_index], "")
        feature_name_unit = f"({feature_name_unit})" if feature_name_unit != "" else "(a.u.)"
        unnormalized_feature_value = single_shap_value.data[shap_feature_name_index]
        normalized_feature_value = (unnormalized_feature_value - X_mean[shap_feature_name_index]) / X_std[shap_feature_name_index]
        feature_text_list.append(f"({normalized_feature_value:5.2f}) {shap_feature_name[shap_feature_name_index]} = {unnormalized_feature_value:.2f} {feature_name_unit}")

    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_xticklabels([])
    # ax.tick_params(axis = 'x', which = 'both', labelbottom = True, labeltop = False)
    # ax.set_xlabel("weoijfo")

    ax_right = ax.twinx()
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_right.tick_params(axis = 'y', which = 'both', right = False)
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks(feature_tick_list)
    ax_right.set_yticklabels(feature_text_list, fontsize = 12)

    ax_upper = ax.twiny()
    ax_upper.spines['top'].set_visible(False)
    ax_upper.spines['right'].set_visible(False)
    ax_upper.spines['bottom'].set_visible(False)
    ax_upper.spines['left'].set_visible(False)
    ax_upper.set_xlim(ax.get_xlim())
    ax_upper.set_xticks([target_property_value])
    normalized_target_property_value = (target_property_value - Y_mean) / Y_std
    ax_upper.set_xticklabels([f"GP prediction ({normalized_target_property_value:.2f})\n{target_property_value:.2f} ({target_property_unit})"], fontsize = 12)
    ax_upper.set_xlabel("")
    ax_upper.get_xaxis().get_majorticklabels()[0].set_ha("left")

    plt.title(f"Sample: {plot_title}", fontsize = 20, pad = 20)
    # plt.tight_layout()
    plt.subplots_adjust(left=0.2, right=0.6, top=0.9, bottom=0.05)
    plt.savefig(plot_filename, transparent = True, bbox_inches='tight')
    plt.close('all')

def plot_shap_all_samples(shap_values, plot_title: str, plot_filename: str, xlabel: str = None):
    import matplotlib.pyplot as plt
    import shap
    shap.plots.beeswarm(shap_values, max_display = 100, show = False)
    plt.title(plot_title, fontsize=20)
    plt.subplots_adjust(left=0.4, right=0.8, top=0.9, bottom=0.1)
    if xlabel:
        plt.xlabel(xlabel)
    plt.savefig(plot_filename, transparent = True)#, bbox_inches='tight')
    plt.close('all')

def plot_gp_results_test_error(gp_index_list: list, 
                               Y_test_err_max_list: list, 
                               Y_test_err_mean_list: list,
                               filename: str,
                               gp_index_list_pareto: list = [],
                               Y_test_err_list_pareto: tuple[list, list] = ([], [])):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import Divider, Size

    figsize = (10, 8)
    margin_ratio = 0.7
    h = [Size.Fixed(1.0), Size.Fixed(figsize[0]*margin_ratio)]
    v = [Size.Fixed(0.7), Size.Fixed(figsize[1]*margin_ratio)]
    fig = plt.figure(figsize = figsize)
    divider = Divider(fig, (0.1, 0.1, 0.9, 0.9), h, v, aspect=False)
    ax = fig.add_axes(divider.get_position(),
                      axes_locator = divider.new_locator(nx=1, ny=1))

    ax.plot(gp_index_list, Y_test_err_max_list, lw = 2, marker = 'o', markersize=5, label=r"Max test error", color = 'r')
    ax.plot(gp_index_list, Y_test_err_mean_list, lw = 2, marker = 'o', markersize=5, label=r"Mean test error", color = 'b')
    if len(Y_test_err_list_pareto[0]) > 0:
        ax.plot(gp_index_list_pareto, Y_test_err_list_pareto[0], lw = 0, marker = 'o', markersize=9, label=r"Pareto Front", color = 'k')
    if len(Y_test_err_list_pareto) > 1 and len(Y_test_err_list_pareto[1]) > 0:
        ax.plot(gp_index_list_pareto, Y_test_err_list_pareto[1], lw = 0, marker = 'o', markersize=9, label=r"Pareto Front", color = 'k')

    ax_upper = ax.twiny()

    ax.legend(title = "", ncol = 1, labelspacing = 0.5, frameon = False, loc = "best")
    # ax.set_yscale('log')
    ax.set_xlabel(r"Training sample size (\#)")
    ax.set_ylabel(r"Test error")

    max_n_sample = gp_index_list[-1]
    upper_xticklabels = [0, 20, 40, 60, 80, 100]
    upper_xticks = [v*0.01*max_n_sample for v in upper_xticklabels]
    for upper_xtick in upper_xticks:
        ax.axvline(x = upper_xtick, color = 'k', ls = ":")

    ax_upper.set_xlim(ax.get_xlim())
    ax_upper.set_xticks(upper_xticks)
    ax_upper.set_xticklabels(upper_xticklabels)
    ax_upper.set_xlabel(r"Training sample size (\%)")

    plt.savefig(filename, dpi = 300, transparent = True)
    plt.close('all')

def plot_gp_results(res: dict, 
                    XY_data_dict: dict, 
                    X_data: np.ndarray, 
                    Y_data: np.ndarray,
                    individual_gp_plot_style: int,
                    target_property: str,
                    filename: str):

    # res["Y_pred_mean"]         
    # res["Y_pred_std"]          
    # res["train_indices"]        
    # res["test_indices"]        
    # res["Y_test_err_max"]      
    # res["Y_test_err_mean"]     
    # res["select_sample_index"] # XY_data_dict

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import Divider, Size

    if individual_gp_plot_style == 0:
        figsize = (10, 10)
    elif individual_gp_plot_style == 1:
        figsize = (10, 8)

    margin_ratio = 0.7
    h = [Size.Fixed(1.0), Size.Fixed(figsize[0]*margin_ratio)]
    v = [Size.Fixed(0.7), Size.Fixed(figsize[1]*margin_ratio)]
    fig = plt.figure(figsize = figsize)
    divider = Divider(fig, (0.1, 0.1, 0.9, 0.9), h, v, aspect=False)
    ax = fig.add_axes(divider.get_position(),
                      axes_locator = divider.new_locator(nx=1, ny=1))
    
    if individual_gp_plot_style == 0:
        max_abs_y = np.abs(Y_data[:, 0]).max()
        xy_ground_truth = np.linspace(-max_abs_y, max_abs_y, 100, endpoint = True)
        ax.plot(xy_ground_truth, xy_ground_truth, lw = 2, color = 'k')

        ax.errorbar(Y_data[res["train_indices"], 0], res["Y_pred_mean"][res["train_indices"]], res["Y_pred_std"][res["train_indices"]], lw = 0, fmt = '-o', elinewidth=2, capsize=3, markersize=9, label=r"GP trained", color = 'red')

        if len(res["test_indices"]) > 0:
            ax.errorbar(Y_data[res["test_indices"], 0], res["Y_pred_mean"][res["test_indices"]], res["Y_pred_std"][res["test_indices"]], lw = 0, fmt = '-o', elinewidth=2, capsize=3, markersize=9, label=r"GP tested", color = 'blue')

        ax.legend(title = "", ncol = 1, labelspacing = 0.5, frameon = False, loc = "best")
        ax.set_xlabel(f"Ground truth (normalized {target_property})")
        ax.set_ylabel(f"GP (normalized {target_property})")

    elif individual_gp_plot_style == 1:
        ax.plot(np.arange(len(Y_data)), Y_data[:, 0], lw = 0, marker = 'o', markersize=9, label=r"Expt.", color = 'k')
        
        ax.errorbar(res["train_indices"], res["Y_pred_mean"][res["train_indices"]], res["Y_pred_std"][res["train_indices"]], lw = 0, fmt = '-o', elinewidth=2, capsize=3, markersize=9, label=r"GP trained", color = 'red')

        ax.errorbar(res["test_indices"], res["Y_pred_mean"][res["test_indices"]], res["Y_pred_std"][res["test_indices"]], lw = 0, fmt = '-o', elinewidth=2, capsize=3, markersize=9, label=r"GP tested", color = 'blue')

        ax.legend(title = "", ncol = 1, labelspacing = 0.5, frameon = False, loc = "best")
        ax.set_xlabel(r"Sample \#")
        ax.set_ylabel(f"Normalized {target_property}")

    plt.savefig(filename, dpi = 300, transparent = True)
    plt.close('all')

def plot_q_data(c: np.ndarray, 
                q: np.ndarray, 
                sq: np.ndarray, 
                data_type: str, 
                filename: str, 
                q_seg_range: np.ndarray = None, 
                sq_seg: np.ndarray = None):
    
    import matplotlib.pyplot as plt
    # from matplotlib import rc
    from mpl_toolkits.axes_grid1 import Divider, Size

    figsize = (10, 8)
    margin_ratio = 0.7
    h = [Size.Fixed(1.0), Size.Fixed(figsize[0]*margin_ratio)]
    v = [Size.Fixed(0.7), Size.Fixed(figsize[1]*margin_ratio)]
    fig = plt.figure(figsize = figsize)
    divider = Divider(fig, (0.1, 0.1, 0.9, 0.9), h, v, aspect=False)
    ax = fig.add_axes(divider.get_position(),
                      axes_locator = divider.new_locator(nx=1, ny=1))
    
    color_series = get_color_series(range(len(c)), 'jet')
    for i_c in range(len(c)):
        non_empty_index = get_non_empty_index(sq[i_c], val_tol = 1e-9)
        ax.plot(q[i_c][non_empty_index], sq[i_c][non_empty_index], label = f"{c[i_c]} m", color = color_series[i_c])

        if q_seg_range is not None:
            q_seg_ = []
            sq_seg_ = []
            n_segs = len(q_seg_range)
            for i_seg in range(n_segs):
                q_seg_.append(q_seg_range[i_seg][0])
                q_seg_.append(q_seg_range[i_seg][1])
                sq_seg_.append(sq_seg[i_c][i_seg])
                sq_seg_.append(sq_seg[i_c][i_seg])
            q_seg_ = np.array(q_seg_)
            sq_seg_ = np.array(sq_seg_)
            ax.plot(q_seg_, sq_seg_, label = f"{c[i_c]} m (segment)", color = color_series[i_c])

    ax.legend(title = "", ncol = 1, labelspacing = 0.5, frameon = False, loc = "best")
    ax.set_xscale('log')
    ax.set_yscale('log')
    q_unit = r"$\mathrm{\AA}$"
    ax.set_xlabel(r'$q$ (%s)'%q_unit)
    ax.set_ylabel(data_type.upper())

    plt.savefig(filename, dpi = 300, transparent = True)
    plt.close('all')
