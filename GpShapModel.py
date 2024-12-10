from _data_preprocess import load_q_data, load_properties, standarize_q_data, get_data_key, fetch_chemical_features_data, get_data_key_c_string, preprocess_data
from _plot import plot_q_data, plot_gp_results, plot_gp_results_test_error, plot_shap_single_sample, plot_shap_all_samples
from _utils import is_pareto
import numpy as np
import copy
import sys
import os
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel

class GSM(object):
    def __init__(self,
                 parm_dict: dict = {}):
        self.gp_dict = parm_dict["gp_parameters"]
        self.data_type_dict = parm_dict["data"]
        self.parm_dict = parm_dict
        self.gpr = None
        # if self.parm_dict["gui"]:
        #     self.create_tk_window()

    #############################################################################################
    #                                           Data                                            #
    #############################################################################################
            
    def load_data(self, overwrite: bool = False, verbose: int = None):
        print("\n\n" + "#"*20 + "Load data" + "#"*20)
        if verbose is None: verbose = self.parm_dict["verbose"]
        session_folder = self.parm_dict["session_folder"]
        only_imide_salt = self.parm_dict.get("only_imide_salt", False)

        self.raw_data_dict              = {}  # data_type -> data
        self.raw_data_key_index_dict    = {}  # data_type -> data_key_index_dict
        self.data_avail_c_dict          = {}  # data_key -> data_type -> c_list
        self.data_avail_c_intersect_map = {}  # data_key -> c_list
        self.data_avail_c_intersect_except_property_map = {} # data_key -> c_list
        self.total_num_chem_dict        = {k: 0 for k in self.data_type_dict.keys()} # data_type -> num_chem
        self.total_num_sample_dict      = {k: 0 for k in self.data_type_dict.keys()} # data_type -> num_sample
        self.all_data_type_key          = 'All'
        self.all_data_type_key_except_property = 'All without property'

        self.total_num_chem_dict[self.all_data_type_key]   = 0
        self.total_num_sample_dict[self.all_data_type_key] = 0
        self.total_num_chem_dict[self.all_data_type_key_except_property]   = 0
        self.total_num_sample_dict[self.all_data_type_key_except_property] = 0

        num_data_type = len(self.data_type_dict.keys())
        for data_type in self.data_type_dict.keys():
            
            data_parm = self.data_type_dict[data_type]

            if data_type == "property": # num_list, name_list, c, dcv
                data, data_key_index_dict = load_properties(data_parm["data_file_path"], session_folder, data_parm["target_property"], data_parm["scaling_factor"], only_imide_salt, overwrite, verbose)
            elif data_type in ["saxs", "ir", "raman"]: # num_list, name_list, c, q, sq
                data, data_key_index_dict = load_q_data(data_parm["data_file_path"], session_folder, data_type, only_imide_salt, overwrite, verbose)
            else:
                raise Exception(f"data type: {data_type} not recongized")

            self.raw_data_dict[data_type] = data
            self.raw_data_key_index_dict[data_type] = data_key_index_dict
            remove_sample_with_q_less_than  = self.data_type_dict[data_type].get("remove_sample_with_q_less_than", -np.inf)
            
            for i_data in range(len(data)):
                num_list, name_list = data[i_data][:2]
                c = data[i_data][2]
                data_key = get_data_key(num_list, name_list)

                if data_type != 'property' and data[i_data][3].max() < remove_sample_with_q_less_than: continue

                if data_key not in self.data_avail_c_dict:
                    self.data_avail_c_dict[data_key] = {k: [] for k in self.data_type_dict.keys()}
                self.data_avail_c_dict[data_key][data_type] = c

        if verbose == 1:
            print()
            print("Check data availability")
        for data_key, data_type_avail_c_dict in self.data_avail_c_dict.items():
            c_intersect = None
            c_intersect_except_property = None
            for data_type, c in data_type_avail_c_dict.items():
                self.total_num_chem_dict[data_type] += (1 if len(c) != 0 else 0)
                self.total_num_sample_dict[data_type] += len(c)
                c_intersect = np.intersect1d(c_intersect, c) if c_intersect is not None else c
                if data_type != "property":
                    c_intersect_except_property = np.intersect1d(c_intersect_except_property, c) if c_intersect_except_property is not None else c
                

            if len(c_intersect) > 0:
                self.data_avail_c_intersect_map[data_key] = c_intersect
                self.total_num_chem_dict[self.all_data_type_key] += 1
                self.total_num_sample_dict[self.all_data_type_key] += len(c_intersect)

                c_intersect_except_property_index = np.logical_not(np.isin(c_intersect_except_property, c_intersect)).nonzero()[0]
                
                if len(c_intersect_except_property_index) > 0:
                    c_intersect_except_property = c_intersect_except_property[c_intersect_except_property_index]
                    self.data_avail_c_intersect_except_property_map[data_key] = c_intersect_except_property
                    self.total_num_chem_dict[self.all_data_type_key_except_property] += 1
                    self.total_num_sample_dict[self.all_data_type_key_except_property] += len(c_intersect_except_property)

        for data_type in self.data_type_dict.keys():
            
            data_parm = self.data_type_dict[data_type]

            if verbose == 1:
                print(data_type.upper() + ('' if data_type != "property" else ' (' + data_parm["target_property"] + ')'))
                print("\tTotal number of chemicals:",  self.total_num_chem_dict[data_type])
                print("\tTotal number of samples:", self.total_num_sample_dict[data_type])
                print()
        
        if verbose == 1:
            print(self.all_data_type_key.capitalize())
            print("\tTotal number of chemicals:",  self.total_num_chem_dict[self.all_data_type_key])
            print("\tTotal number of samples:", self.total_num_sample_dict[self.all_data_type_key])
            print()
            print("List of data with complete availability")
            for data_key, c in self.data_avail_c_intersect_map.items():
                print(f"\t{data_key}: ", *c)

            print(self.all_data_type_key_except_property.capitalize())
            print("\tTotal number of chemicals:",  self.total_num_chem_dict[self.all_data_type_key_except_property])
            print("\tTotal number of samples:", self.total_num_sample_dict[self.all_data_type_key_except_property])
            print()
            print("List of data with complete availability except property")
            for data_key, c in self.data_avail_c_intersect_except_property_map.items():
                print(f"\t{data_key}: ", *c)

    def preprocess_data(self, overwrite: bool = False, verbose: int = None):
        print("\n\n" + "#"*20 + "Preprocess data" + "#"*20)
        if verbose is None: verbose = self.parm_dict["verbose"]
        session_folder = self.parm_dict["session_folder"]
        
        if os.path.isfile(f"{session_folder}/pp.pickle") and overwrite is False:
            print(f"Preprocessed data exists, skip loading raw data.")
            with open(f"{session_folder}/pp.pickle", "rb") as f:
                self.processed_data_dict , \
                self.feature_name_list   , \
                self.X_data              , \
                self.Y_data              , \
                self.XY_data_dict        , \
                self.num_samples         , \
                self.num_features        , \
                self.X_mean              , \
                self.X_std               , \
                self.Y_mean              , \
                self.Y_std                   = pickle.load(f)
        else:
            self.processed_data_dict = {} # data_type -> data_key -> [c, q, sq, q_seg_range, sq_seg]
                                          #                       -> [c, p, feature, feature_name]
            self.feature_name_list = []
            self.X_data = []
            self.Y_data = []
            self.XY_data_dict = [] # X_data/Y_data index -> (data_key, c)
            self.num_samples  = 0.
            self.num_features = 0.
            self.X_mean = []
            self.X_std  = []
            self.Y_mean = 0.
            self.Y_std  = 0.

            self.processed_data_dict , \
            self.feature_name_list   , \
            self.X_data              , \
            self.Y_data              , \
            self.XY_data_dict        , \
            self.num_samples         , \
            self.num_features        , \
            self.X_mean              , \
            self.X_std               , \
            self.Y_mean              , \
            self.Y_std  = preprocess_data(self.data_type_dict, self.data_avail_c_intersect_map, self.raw_data_dict, self.raw_data_key_index_dict, True, verbose)

            # for data_type in self.data_type_dict.keys():
                
            #     if data_type == "property": continue

            #     if verbose == 1:
            #         print(f"Preprocessing {data_type.upper()} data")
            #     data_parm = self.data_type_dict[data_type]

            #     self.processed_data_dict[data_type] = standarize_q_data(self.data_avail_c_intersect_map, self.raw_data_dict[data_type], self.raw_data_key_index_dict[data_type], data_parm, verbose)

            # data_type = "property"
            # data_parm = self.data_type_dict[data_type]
            # self.processed_data_dict[data_type] = fetch_chemical_features_data(self.data_avail_c_intersect_map, self.raw_data_dict[data_type], self.raw_data_key_index_dict[data_type], data_parm, verbose)

            # for data_key in self.data_avail_c_intersect_map.keys():
            #     target_c = self.data_avail_c_intersect_map[data_key]
                
            #     data_type = "property"
            #     c, p, feature, feature_name = self.processed_data_dict[data_type][data_key]
            #     c_idx = np.isin(c, target_c).nonzero()[0]

            #     assert len(target_c) == len(c_idx), f"Concentration missing, target: {target_c}, avail: {c[c_idx]}"

            #     for i_c in range(len(c_idx)):
            #         self.X_data.append(np.expand_dims(np.concatenate((c[c_idx[i_c] : c_idx[i_c] + 1], np.copy(feature))), 0))
            #         self.Y_data.append(np.expand_dims(p[c_idx[i_c] : c_idx[i_c] + 1], 0))
            #         self.XY_data_dict.append((data_key, c[c_idx[i_c]]))

            #     for data_type in self.data_type_dict.keys():

            #         if data_type == "property": continue

            #         c, q, sq, q_seg_range, sq_seg = self.processed_data_dict[data_type][data_key]
            #         c_idx = np.isin(c, target_c).nonzero()[0]

            #         assert len(target_c) == len(c_idx), f"Concentration missing, target: {target_c}, avail: {c[c_idx]}"

            #         for i_c in range(len(c_idx)):
            #             self.X_data[-len(c_idx) + i_c] = np.concatenate((self.X_data[-len(c_idx) + i_c], sq_seg[c_idx[i_c] : c_idx[i_c] + 1]), axis = -1)

            # self.feature_name_list = ["concentration"] + feature_name
            # for data_type in self.data_type_dict.keys():
            #     if data_type == "property": continue
            #     self.feature_name_list += [f"{data_type} ({v[0]:.2f} ~ {v[1]:.2f})" for v in  q_seg_range]

            # self.X_data = np.concatenate(self.X_data)
            # self.Y_data = np.concatenate(self.Y_data)
            # self.num_samples = len(self.X_data)
            # self.num_features = self.X_data.shape[1]

            # self.Y_mean = self.Y_data.mean()
            # self.Y_std = self.Y_data.std(ddof = 1)
            # for i_feature in range(self.num_features):
            #     xtmp = self.X_data[:, i_feature]
            #     xtmp = xtmp[np.logical_not(np.isnan(xtmp))]
            #     self.X_mean += [xtmp.mean()]
            #     self.X_std += [xtmp.std(ddof = 1)]
            # self.X_mean = np.array(self.X_mean)
            # self.X_std = np.array(self.X_std)

            # # i_feature = 0
            # # j_feature = 0
            # # while j_feature < self.num_features:
            # #     if self.feature_name_list[i_feature] == self.feature_name_list[j_feature]:
            # #         j_feature += 1
            # #         continue
            # #     else:
            # #         xtmp = self.X_data[:, i_feature : j_feature]
            # #         xtmp = xtmp[np.logical_not(np.isnan(xtmp))]
            # #         self.X_mean += [xtmp.mean()] * (j_feature - i_feature)
            # #         self.X_std += [xtmp.std(ddof = 1)] * (j_feature - i_feature)
            # #         i_feature = j_feature

            # # xtmp = self.X_data[:, i_feature : j_feature]
            # # xtmp = xtmp[np.logical_not(np.isnan(xtmp))]
            # # self.X_mean += [xtmp.mean()] * (j_feature - i_feature)
            # # self.X_std += [xtmp.std(ddof = 1)] * (j_feature - i_feature)
            # # self.X_mean = np.array(self.X_mean)
            # # self.X_std = np.array(self.X_std)

            # duplicate_feature_index = 0
            # for i_feature in range(self.num_features - 1):
            #     if self.feature_name_list[i_feature] == self.feature_name_list[i_feature + 1]:
            #         duplicate_feature_index += 1
            #         self.feature_name_list[i_feature] += f" {duplicate_feature_index}"
            #     elif duplicate_feature_index > 0:
            #         duplicate_feature_index += 1
            #         self.feature_name_list[i_feature] += f" {duplicate_feature_index}"
            #         duplicate_feature_index = 0
            # if duplicate_feature_index > 0:
            #     duplicate_feature_index += 1
            #     self.feature_name_list[-1] += f" {duplicate_feature_index}"

            # for i_feature in range(self.num_features):
            #     nan_index = np.isnan(self.X_data[:, i_feature])
            #     if len(nan_index.nonzero()[0]) > 0:
            #         self.X_data[nan_index, i_feature] = self.X_mean[i_feature]
            
            # zero_index = (self.X_std == 0).nonzero()[0]
            # if len(zero_index) > 0:
            #     print("warning: these features have zero std: ", *zero_index)
            #     print("they are set to unity")
            #     self.X_std[zero_index] = 1.0

            self.X_data = (self.X_data - self.X_mean) / self.X_std
            self.Y_data = (self.Y_data - self.Y_mean) / self.Y_std

            with open(f"{session_folder}/pp.pickle", "wb") as f:
                pickle.dump((self.processed_data_dict , 
                            self.feature_name_list   , 
                            self.X_data              , 
                            self.Y_data              , 
                            self.XY_data_dict        , 
                            self.num_samples         , 
                            self.num_features        , 
                            self.X_mean              , 
                            self.X_std               , 
                            self.Y_mean              , 
                            self.Y_std                 ), f)
            
        if verbose == 1:
            print()
            print(f"Number of samples: {self.num_samples}")
            print(f"Number of features: {self.num_features}")
            print(f"Feature mean/std:")
            for i_feature, a_feature_name in enumerate(self.feature_name_list):
                m = self.X_mean[i_feature]
                s = self.X_std[i_feature]
                print(f"\t{i_feature:5d}:\t{a_feature_name:35s}\t{m:.2e} +- {s:.2e}")
            print(f"Property mean/std:")
            target_property = self.data_type_dict["property"]["target_property"]
            print(f"\t      \t{target_property:35s}\t{self.Y_mean:.2e} +- {self.Y_std:.2e}")

    #############################################################################################
    #                                             GP                                            #
    #############################################################################################
            
    def create_gp_model(self):
        gp_kernel_scale = self.gp_dict.get("gp_kernel_scale", 1.0)
        gp_length_scale = self.gp_dict.get("gp_length_scale", 1.0)
        intrinsic_noise = self.gp_dict.get("intrinsic_noise", 1.0)
        kernel = gp_kernel_scale * RBF(length_scale=gp_length_scale, length_scale_bounds=(1e-2, 1e4))
        # kernel = gp_kernel_scale * RationalQuadratic(length_scale=gp_length_scale, alpha=intrinsic_noise**2, length_scale_bounds=(1e-2, 1e4), alpha_bounds=(1e-05, 100000.0))
        self.gpr = GaussianProcessRegressor(kernel=kernel, alpha=intrinsic_noise**2)

    def load_gp_model(self, gpr_filename: str):
        with open(gpr_filename, "rb") as f:
            self.gpr, self.gp_train_history_dict = pickle.load(f)
        
    def save_gp_model(self, gpr_filename: str):
        with open(gpr_filename, "wb") as f:
            pickle.dump((self.gpr, self.gp_train_history_dict), f)
            
    def gp_train(self, overwrite: bool = False, verbose: int = None):
        print("\n\n" + "#"*20 + "GP training" + "#"*20)
        session_folder = self.parm_dict["session_folder"]
        if verbose is None: verbose = self.parm_dict["verbose"]
        random_seed = self.gp_dict["random_seed"]

        gpr_filename = f"{session_folder}/gpr_train.pickle"

        if os.path.isfile(gpr_filename) and overwrite is False:
            self.load_gp_model(gpr_filename)
            print("GP model already exists, load GP model and skip training.")
            return
        else:
            print("Create new GP model and start training")
            self.create_gp_model()

        self.gp_train_history_dict = {} # gp_index -> train history

        number_of_initial_training_samples = self.gp_dict["number_of_initial_training_samples"]
        sample_indices = np.arange(self.num_samples)
        np.random.seed(random_seed)
        np.random.shuffle(sample_indices)

        train_indices = sample_indices[:number_of_initial_training_samples]
        test_indices = sample_indices[number_of_initial_training_samples:]

        if verbose == 1:
            print()
            print("Initial sample for GP training:")
            for train_index in train_indices:
                data_key, c = self.XY_data_dict[train_index]
                print(f"\t{data_key}: {c}")

        X_train = self.X_data[train_indices]
        Y_train = self.Y_data[train_indices]

        for i_sample in range(number_of_initial_training_samples, self.num_samples + 1):

            self.gpr.fit(X_train, Y_train)

            Y_pred_mean, Y_pred_std = self.gpr.predict(self.X_data, return_std=True)

            if len(test_indices) > 0:
                Y_test_mean, Y_test_std = Y_pred_mean[test_indices], Y_pred_std[test_indices]
                max_Y_std_index = np.argmax(Y_test_std)
                Y_test_err = np.absolute(Y_test_mean - self.Y_data[test_indices, 0])
                Y_test_std_max = Y_test_std.max()
                Y_test_err_max = Y_test_err.max()
                Y_test_err_mean = Y_test_err.mean()
                select_sample_index = test_indices[max_Y_std_index]
            else:
                Y_test_std_max = 0.
                Y_test_err_max = 0.
                Y_test_err_mean = 0.
                select_sample_index = -1

            self.gp_train_history_dict[i_sample]                        = {}
            self.gp_train_history_dict[i_sample]["gpr"]                 = copy.deepcopy(self.gpr)
            self.gp_train_history_dict[i_sample]["Y_pred_mean"]         = np.copy(Y_pred_mean)
            self.gp_train_history_dict[i_sample]["Y_pred_std"]          = np.copy(Y_pred_std)
            self.gp_train_history_dict[i_sample]["train_indices"]       = np.copy(train_indices)
            self.gp_train_history_dict[i_sample]["test_indices"]        = np.copy(test_indices)
            self.gp_train_history_dict[i_sample]["Y_test_std_max"]      = Y_test_std_max
            self.gp_train_history_dict[i_sample]["Y_test_err_max"]      = Y_test_err_max
            self.gp_train_history_dict[i_sample]["Y_test_err_mean"]     = Y_test_err_mean
            self.gp_train_history_dict[i_sample]["select_sample_index"] = select_sample_index

            if verbose == 1:
                Y_test_err_max_print = self.gp_train_history_dict[i_sample]["Y_test_err_max"]
                Y_test_err_mean_print = self.gp_train_history_dict[i_sample]["Y_test_err_mean"]
                print(f"{i_sample:5d}\tMax Error: {Y_test_err_max_print:.2e}\tMean Error: {Y_test_err_mean_print:.2e}")
                data_key, c = self.XY_data_dict[select_sample_index]
                print(f"Select sample for next iteration: {data_key}: {c}")
                print()
            
            if len(test_indices) == 0: break

            X_train = np.vstack((X_train, self.X_data[test_indices][max_Y_std_index:max_Y_std_index+1]))
            Y_train = np.vstack((Y_train, self.Y_data[test_indices][max_Y_std_index:max_Y_std_index+1]))
            train_indices = np.append(train_indices, test_indices[max_Y_std_index])
            test_indices = np.delete(test_indices, max_Y_std_index)

        self.save_gp_model(gpr_filename)

    def gp_suggest(self, verbose: int = None):
        print("\n\n" + "#"*20 + "Suggesting new measurments" + "#"*20)
        if verbose is None: verbose = self.parm_dict["verbose"]
        session_folder = self.parm_dict["session_folder"]
        # predict GP mean/std from chemical systems where saxs, ir, and raman are avail except property

        data_avail_c_intersect_except_property_map = copy.deepcopy(self.data_avail_c_intersect_except_property_map)
        fix_q_range = {}
        data_with_not_enough_q_range = set()
        for data_type in self.data_type_dict.keys():

            if data_type == "property": continue
            for data_key, c in self.data_avail_c_intersect_map.items():
                c, q, sq, q_seg_range, sq_seg = self.processed_data_dict[data_type][data_key]
                fix_q_range[data_type] = [q_seg_range[0][0], q_seg_range[-1][-1]]
                break
            
            data = self.raw_data_dict[data_type]
            data_key_index_dict = self.raw_data_key_index_dict[data_type]
            for data_key, c in data_avail_c_intersect_except_property_map.items():
                data_key_index = data_key_index_dict[data_key]
                _, _, cc, q, _ = data[data_key_index]

                if 0.0 in c:
                    data_avail_c_intersect_except_property_map[data_key] = np.delete(data_avail_c_intersect_except_property_map[data_key], (c == 0.0).nonzero()[0])
                    if len(data_avail_c_intersect_except_property_map[data_key]) == 0:
                        data_with_not_enough_q_range.add(data_key)
                        print(f"warning: removing {data_key} as it only contains zero concentration data.")

                if q.min() > fix_q_range[data_type][0] or q.max() < fix_q_range[data_type][1]:
                    print(f"warning: removing {data_key} from suggestiong as its q range in {data_type} is not enough for infernece")
                    data_with_not_enough_q_range.add(data_key)

        if len(data_with_not_enough_q_range) > 0:
            for data_key in data_with_not_enough_q_range:
                data_avail_c_intersect_except_property_map.pop(data_key)

        # for data_key, c in data_avail_c_intersect_except_property_map.items():
        #     print(data_key, c)

        processed_data_dict , \
        feature_name_list   , \
        X_data              , \
        Y_data              , \
        XY_data_dict        , \
        num_samples         , \
        num_features        , \
        X_mean              , \
        X_std               , \
        Y_mean              , \
        Y_std  = preprocess_data(self.data_type_dict, data_avail_c_intersect_except_property_map, self.raw_data_dict, self.raw_data_key_index_dict, False, verbose)

        X_data = (X_data - self.X_mean) / self.X_std

        Y_pred_mean, Y_pred_std = self.gpr.predict(X_data, return_std=True)

        Y_pred_mean = Y_pred_mean * self.Y_std + self.Y_mean
        Y_pred_std *= self.Y_std

        Y_argsort = np.argsort(Y_pred_mean)

        print()
        print("List of sorted GP prediction sorting in acending order")
        for i_sort_sample in Y_argsort:
            i_sample = Y_argsort[i_sort_sample]
            data_key, c = XY_data_dict[i_sort_sample]

            plot_title    = get_data_key_c_string(data_key.replace("-", "/"), c).replace("_", " ").replace("-", ":").replace("/", "-")
            plot_title_list = plot_title.split(" ")
            plot_title    = f"{plot_title_list[0]} ({plot_title_list[1]}) [{plot_title_list[2]}]"
                
            print(f"{i_sample:5d}: {plot_title:45s} -> {Y_pred_mean[i_sort_sample]:.2f} +- {Y_pred_std[i_sort_sample]:.3f}")

    #############################################################################################
    #                                         SHAP                                              #
    #############################################################################################
        
    def shap_analysis(self, gp_index = -1, overwrite: bool = False, verbose: int = None):
        print("\n\n" + "#"*20 + "Shapley analysis" + "#"*20)
        if verbose is None: verbose = self.parm_dict["verbose"]
        session_folder = self.parm_dict["session_folder"]
        import shap
        import pandas as pd

        if os.path.isfile(f"{session_folder}/shap_analysis.pickle") and overwrite is False:
            print("Shapley analysis results already exist, load results and skip analysis.")
            with open(f"{session_folder}/shap_analysis.pickle", "rb") as f:
                self.shap_feature_name, self.shap_values_dict = pickle.load(f)
        else:
            self.shap_feature_name = [v.replace('~', '-') for v in self.feature_name_list]
            self.shap_values_dict = {}
            last_gp_index = max(list(self.gp_train_history_dict.keys()))
            gp_x_data     = np.copy(self.gp_train_history_dict[last_gp_index]["gpr"].X_train_) * self.X_std + self.X_mean

            gp_index_list        = []
            if gp_index == -1:
                for i_sample in self.gp_train_history_dict.keys():
                    gp_index_list        .append(i_sample)
            elif isinstance(gp_index, list):
                gp_index_list = [v for v in gp_index if self.gp_train_history_dict.get(v)]
            elif self.gp_train_history_dict.get(gp_index):
                gp_index_list        = [gp_index]
            else:
                print("GP index not found.")
                return
            
            for i_gp_index in range(len(gp_index_list)):
                # gp_x_data = np.copy(gpr.X_train_) * self.X_std + self.X_mean
                gp_index = gp_index_list[i_gp_index]
                if verbose == 1 : print(f"Analyzing GP({gp_index}) model...")
                gpr = self.gp_train_history_dict[gp_index]['gpr']
                gpr_for_shap = lambda x: gpr.predict((x - self.X_mean) / self.X_std) * self.Y_std + self.Y_mean
                dataset = pd.DataFrame({self.shap_feature_name[v]: gp_x_data[:, v] for v in range(self.num_features)})
                custom_masker = lambda mask, x: (x * mask).reshap(1, len(x))
                custom_masker = gp_x_data
                explainer = shap.Explainer(gpr_for_shap, masker = custom_masker)
                # explainer = shap.Explainer(gpr_for_shap, masker = custom_masker, algorithm="exact")
                shap_values = explainer(dataset)

                self.shap_values_dict[gp_index] = copy.deepcopy(shap_values)

            with open(f"{session_folder}/shap_analysis.pickle", "wb") as f:
                pickle.dump((self.shap_feature_name, self.shap_values_dict), f)

    #############################################################################################
    #                                       Plotting                                            #
    #############################################################################################

    def plot_raw_data(self, data_type: str, plot_extension: str = "png", overwrite: bool = False, verbose: int = None):
        session_folder = self.parm_dict["session_folder"]
        if verbose is None: verbose = self.parm_dict["verbose"]

        sub_folder = f"raw_{data_type}"
        os.makedirs(f"{session_folder}/{sub_folder}", exist_ok = True)
        session_folder = self.parm_dict["session_folder"]
        raw_data_dict = self.raw_data_dict[data_type]
        if len(raw_data_dict) == 0:
            print("No raw data to plot")
            return
        
        raw_data_key_index_dict = self.raw_data_key_index_dict[data_type]
        for data_key in self.data_avail_c_intersect_map.keys():
            filesufix = get_data_key_c_string(data_key, -1)
            plot_filename = f"{session_folder}/{sub_folder}/{data_type}_{filesufix}.{plot_extension}"
            if os.path.isfile(plot_filename) is False or overwrite is True:
                if verbose == 1: print(f"Plotting {plot_filename}")
                data_key_index = raw_data_key_index_dict[data_key]
                _, _, c, q, sq = raw_data_dict[data_key_index]
                plot_q_data(c, q, sq, data_type, plot_filename)

    def plot_processed_data(self, data_type: str, plot_extension: str = "png", overwrite: bool = False, verbose: int = None):
        session_folder = self.parm_dict["session_folder"]
        if verbose is None: verbose = self.parm_dict["verbose"]

        sub_folder = f"preprocessed_{data_type}"
        os.makedirs(f"{session_folder}/{sub_folder}", exist_ok = True)

        processed_data_dict = self.processed_data_dict[data_type]
        if len(processed_data_dict) == 0:
            print("No processed data to plot")
            return
        
        for data_key in processed_data_dict.keys():
            filesufix = get_data_key_c_string(data_key, -1)
            plot_filename = f"{session_folder}/{sub_folder}/{data_type}_{filesufix}.{plot_extension}"
            if os.path.isfile(plot_filename) is False or overwrite is True:
                c, q, sq, q_seg_range, sq_seg = processed_data_dict[data_key]
                if verbose == 1: print(f"Plotting {plot_filename}")
                plot_q_data(c, q, sq, data_type, plot_filename, q_seg_range, sq_seg)

    def plot_gp_results(self, plot_pareto: int = 0, plot_every: int = 1, individual_gp_plot_style: int = 0, plot_extension: str = "png", overwrite: bool = False, verbose: int = None):
        """
            plot the following figures:
                test error
                individual GP results for each number of trained samples
            args:
                plot_pareto: 0 = don't plot pareto front
                             1 = plot pareto front based on max test error
                             2 = plot pareto front based on mean test error
                             3 = plot pareto front based on both max and mean test error
                plot_every:  0 = don't plot individual GP results
                            >1 = plot every plot_every individual GP results (always includes the first and the last)        
        """
        session_folder = self.parm_dict["session_folder"]
        if verbose is None: verbose = self.parm_dict["verbose"]

        sub_folder = "single_gp_result"
        os.makedirs(f"{session_folder}/{sub_folder}", exist_ok = True)
            
        gp_index_list        = []
        Y_test_std_max_list  = []
        Y_test_err_max_list  = []
        Y_test_err_mean_list = []
        for i_sample in self.gp_train_history_dict.keys():
            gp_index_list       .append(i_sample)
            Y_test_std_max_list .append(self.gp_train_history_dict[i_sample]["Y_test_std_max"] )
            Y_test_err_max_list .append(self.gp_train_history_dict[i_sample]["Y_test_err_max"] )
            Y_test_err_mean_list.append(self.gp_train_history_dict[i_sample]["Y_test_err_mean"])

        gp_index_list_to_plot = list(self.gp_train_history_dict.keys())
        if plot_pareto == 0:
            assert plot_every < len(gp_index_list_to_plot), f"plot_every ({plot_every}) should be less than the number of samples ({len(gp_index_list_to_plot)})"
            if plot_every > 0:
                if gp_index_list_to_plot[-1] not in gp_index_list_to_plot[::plot_every]:
                    gp_index_list_to_plot = gp_index_list_to_plot[::plot_every] + [gp_index_list_to_plot[-1]]
                else:
                    gp_index_list_to_plot = gp_index_list_to_plot[::plot_every]
        else:
            if plot_pareto == 1:
                Y_test_err_list_pareto = (Y_test_err_max_list, )
            elif plot_pareto == 2:
                Y_test_err_list_pareto = (Y_test_err_mean_list, )
            elif plot_pareto == 3:
                Y_test_err_list_pareto = (Y_test_err_max_list, Y_test_err_mean_list)

            is_efficient_index = is_pareto(np.vstack((gp_index_list, *Y_test_err_list_pareto)).transpose()).nonzero()[0]
            gp_index_list_pareto = [gp_index_list[v] for v in is_efficient_index]
            Y_test_err_list_pareto = tuple([[Y_test_err_list_pareto[iv][v] for v in is_efficient_index] for iv in range(len(Y_test_err_list_pareto))])
            gp_index_list_to_plot = gp_index_list_pareto

            if verbose == 1:
                print("Results for Pareto Front:")
                for i_sample in range(len(gp_index_list_pareto)):
                    p = gp_index_list_pareto[i_sample] / max(gp_index_list) * 100
                    print(f"\t{gp_index_list_pareto[i_sample]:5d}\t{p:.2f}\t", end = '')
                    for iv in range(len(Y_test_err_list_pareto)):
                        print(f"{Y_test_err_list_pareto[iv][i_sample]:.2e}\t", end = '')
                    print()
        
        plot_filename = f"{session_folder}/gp_test_error.{plot_extension}"
        if os.path.isfile(plot_filename) is False or overwrite is True:
            if verbose == 1: print(f"Plotting {plot_filename}")
            plot_gp_results_test_error(gp_index_list, 
                                       Y_test_err_max_list, 
                                       Y_test_err_mean_list, 
                                       plot_filename,
                                       gp_index_list_pareto   if plot_pareto > 0 else [],
                                       Y_test_err_list_pareto if plot_pareto > 0 else [])
        if plot_every > 0:
            if verbose == 1: print(f"Plotting individual GP results")
            for i_sample in gp_index_list_to_plot:
                res = self.gp_train_history_dict[i_sample]
                plot_filename = f"{session_folder}/{sub_folder}/gp_{i_sample}.{plot_extension}"
                if os.path.isfile(plot_filename) is False or overwrite is True:
                    if verbose == 1: print(f"Plotting {plot_filename}")
                    target_property = self.data_type_dict["property"]["target_property"]
                    plot_gp_results(res, self.XY_data_dict, self.X_data, self.Y_data, individual_gp_plot_style, target_property, plot_filename)

    def plot_shap_results(self, plot_extension: str = "png", overwrite: bool = False, verbose: int = None):
        session_folder = self.parm_dict["session_folder"]
        if verbose is None: verbose = self.parm_dict["verbose"]

        sub_folder = "single_shap_result"
        os.makedirs(f"{session_folder}/{sub_folder}", exist_ok = True)
        # self.shap_feature_name
        last_gp_index     = max(list(self.gp_train_history_dict.keys()))
        gpr_train_indices = self.gp_train_history_dict[last_gp_index]["train_indices"]

        for gp_index in self.shap_values_dict.keys():
            shap_values = self.shap_values_dict[gp_index]

            plot_title    = ""
            plot_filename = f"{session_folder}/shap_all_gp_{gp_index}.{plot_extension}"
            if os.path.isfile(plot_filename) is False or overwrite is True:
                if verbose == 1: print(f"Plotting {plot_filename}")
                target_property = self.data_type_dict["property"]["target_property"]
                target_property_unit = self.data_type_dict["property"]["target_property_unit_for_plot"]
                xlabel = f"SHAP value for {target_property} ({target_property_unit})"
                plot_shap_all_samples(shap_values, plot_title, plot_filename, xlabel)

            for i_shap_index in range(len(shap_values)):
                i_train_index = gpr_train_indices[i_shap_index]
                # shap_rank_index = np.argsort(np.abs(shap_values[i_shap_index].values))
                # shap_v = shap_values[i_shap_index].values[shap_rank_index]
                # shap_d = shap_values[i_shap_index].data[shap_rank_index]

                data_key, c   = self.XY_data_dict[i_train_index]
                filesufix     = get_data_key_c_string(data_key, c)
                plot_title    = get_data_key_c_string(data_key.replace("-", "/"), c).replace("_", " ").replace("-", ":").replace("/", "-")
                plot_title_list = plot_title.split(" ")
                plot_title    = f"{plot_title_list[0]} ({plot_title_list[1]}) [{plot_title_list[2]}]"
                plot_filename = f"{session_folder}/{sub_folder}/shap_{i_shap_index}_gp_{gp_index}_{filesufix}.{plot_extension}"

                if os.path.isfile(plot_filename) is False or overwrite is True:
                    if verbose == 1: print(f"Plotting {plot_filename}")
                    target_property = self.data_type_dict["property"]["target_property"]
                    target_property_unit = self.data_type_dict["property"]["target_property_unit_for_plot"]
                    plot_shap_single_sample(shap_values[i_shap_index], plot_title, plot_filename, target_property, target_property_unit, self.X_mean, self.X_std, self.Y_mean, self.Y_std, self.shap_feature_name)
    
    # def create_tk_window(self):
    #     import tkinter as tk
    #     self.gui_dict = {}
    #     self.gui_dict["window"] = tk.Tk()
    #     window_size = (800, 600)
    #     self.gui_dict["window"].title("GpShapModel")
    #     self.gui_dict["window"].geometry(f"{window_size[0]}x{window_size[1]}")
    #     self.gui_dict["window"].config(bg='black')
    #     self.gui_dict["textbox"] = tk.Text(self.gui_dict["window"], 
    #                                        width  = int(window_size[0]*0.1),
    #                                        height = int(window_size[1]*0.1),
    #                                        bg="black",
    #                                        fg="white")
    #     self.gui_dict["textbox"].pack(side = tk.LEFT, padx = 20, pady = 20)
    #     sys.stdout = RedirectOutput(self.gui_dict["textbox"])
    #     self.gui_dict[""]
    #     self.gui_dict["window"].after(1000, self.load_data)
    #     self.gui_dict["window"].mainloop()



# class RedirectOutput:
#     def __init__(self, text_widget):
#         import tkinter as tk
#         self.text_widget = text_widget
#         self.text_widget.config(state=tk.NORMAL)
    
#     def write(self, message):
#         import tkinter as tk
#         self.text_widget.insert(tk.END, message)
#         self.text_widget.see(tk.END)
    
#     def flush(self):
#         pass
