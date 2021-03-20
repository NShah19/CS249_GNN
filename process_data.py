import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
import os
import json
import nilearn.connectome


DATA_DIR = "ABIDE_data\\Outputs\\cpac\\filt_global\\rois_aal"
SAVE_DIR = "ABIDE_data"
LABELS_PATH = "ABIDE_data\\Phenotypic_V1_0b_preprocessed1.csv"

os.makedirs(os.path.join(SAVE_DIR, "raw"), exist_ok=True)

all_save_data = {}
filenames = [fn for fn in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, fn))]
for idx, filename in enumerate(filenames):
	print('progress:', "{:.2%}".format(idx/len(filenames)))
	data_path = os.path.join(DATA_DIR, filename)

	func_parc = pd.read_csv(data_path, sep="\t")

	# tseries_zscore = sp.stats.zscore(func_parc, axis=1)
	# print(tseries_zscore.shape)

	# corr_mat = np.corrcoef(func_parc, rowvar=False)
	# pcorr_mat = -np.linalg.inv(corr_mat)

	corr_measure = nilearn.connectome.ConnectivityMeasure(kind='correlation')
	corr_mat = corr_measure.fit_transform([func_parc.to_numpy()])[0]
	pcorr_measure = nilearn.connectome.ConnectivityMeasure(kind='partial correlation')
	pcorr_mat = pcorr_measure.fit_transform([func_parc.to_numpy()])[0]

	for i in range(len(corr_mat)):
		corr_mat[i][i] = 0
		pcorr_mat[i][i] = 0

	# corr_mat = np.nan_to_num(corr_mat)
	# pcorr_mat = np.nan_to_num(pcorr_mat)

	pheno_info = pd.read_csv(LABELS_PATH)
	file_id = filename.partition("_rois")[0]
	diagnosis = pheno_info[pheno_info["FILE_ID"] == file_id]["DX_GROUP"]

	save_data = {}
	save_data["corr"] = corr_mat.tolist()
	save_data["pcorr"] = pcorr_mat.tolist()
	save_data["indicator"] = int(diagnosis) - 1
	all_save_data[filename] = save_data

	if not np.any(np.isnan(corr_mat)):
		save_path = os.path.join(SAVE_DIR, "raw", os.path.splitext(filename)[0] + ".txt")
		with open(save_path, "w") as f:
			json.dump(save_data, f, indent=4)
	else:
		print("NANs")
