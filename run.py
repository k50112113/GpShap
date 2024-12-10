from GpShapModel import GSM
import yaml
import sys
import os

plot_extension = "pdf"
yaml_filename  = sys.argv[1]

with open(sys.argv[1]) as f:
    yaml_file = yaml.load(f, Loader=yaml.FullLoader)

session_folder = yaml_file.get("session_folder")
os.makedirs(session_folder, exist_ok = True)
gsm_model = GSM(yaml_file)

gsm_model.load_data(overwrite = False, verbose = 1)
# gsm_model.plot_raw_data(data_type = "saxs", plot_extension = plot_extension, overwrite = False, verbose = 1)

gsm_model.preprocess_data(overwrite = False, verbose = 1)
# gsm_model.plot_processed_data(data_type = "saxs", plot_extension = plot_extension, overwrite = False, verbose = 1)


gsm_model.gp_train(overwrite = False, verbose = 0)
gsm_model.plot_gp_results(plot_pareto = 1, plot_every = 1, individual_gp_plot_style = 0, plot_extension = plot_extension, overwrite = False, verbose = 1)

gsm_model.gp_suggest(verbose = 1)
exit()

if "ver1" in yaml_filename:
    gsm_model.shap_analysis(gp_index = [114, 138], overwrite = False, verbose = 1)
elif "ver2" in yaml_filename:
    gsm_model.shap_analysis(gp_index = [116, 138], overwrite = False, verbose = 1)
elif "ver3" in yaml_filename:
    gsm_model.shap_analysis(gp_index = [121, 138], overwrite = False, verbose = 1)
elif "ver4" in yaml_filename:
    gsm_model.shap_analysis(gp_index = [87, 104], overwrite = False, verbose = 1)
elif "ver5" in yaml_filename:
    gsm_model.shap_analysis(gp_index = [291, 301], overwrite = False, verbose = 1)
elif "ver6" in yaml_filename:
    gsm_model.shap_analysis(gp_index = [340, 379], overwrite = False, verbose = 1)
elif "ver7" in yaml_filename:
    gsm_model.shap_analysis(gp_index = [179, 197], overwrite = False, verbose = 1)
elif "ver8" in yaml_filename:
    gsm_model.shap_analysis(gp_index = [116, 138], overwrite = False, verbose = 1)

gsm_model.plot_shap_results(plot_extension = plot_extension, overwrite = True, verbose = 1)