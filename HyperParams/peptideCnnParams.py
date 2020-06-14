#for peptideCNN

#batch sizes
hyper_params1 = {}
hyper_params1['name'] = "peptideCNN2"
hyper_params1['batch_size'] = 16
hyper_params1['data_path'] = "C:\\Users\\diash\\repos\\DeepRTplus\\data\\ATLANTIS_SILICA.txt"
hyper_params1['epochs'] = 25
hyper_params1['length_peptide_sequence'] = 25
hyper_params1['number_of_filters'] = 160
hyper_params1['kernel_sizes'] = [7, 5]
hyper_params1['padding_sizes'] = [5, 3]
hyper_params1['dropout_rate'] = 0.05
hyper_params1['peptide_input_dim'] = 20
hyper_params1['learning_rate'] = 0.0001
hyper_params1['backwards'] = False

hyper_params2 = {}
hyper_params2['name'] = "peptideCNN2"
hyper_params2['batch_size'] = 32
hyper_params2['data_path'] = "C:\\Users\\diash\\repos\\DeepRTplus\\data\\ATLANTIS_SILICA.txt"
hyper_params2['epochs'] = 20
hyper_params2['length_peptide_sequence'] = 25
hyper_params2['number_of_filters'] = 40
hyper_params2['kernel_sizes'] = [5, 3]
hyper_params2['padding_sizes'] = [3, 2]
hyper_params2['dropout_rate'] = 0.1
hyper_params2['peptide_input_dim'] = 20
hyper_params2['learning_rate'] = 0.00001
hyper_params2['backwards'] = True

hyper_params3 = {}
hyper_params3['name'] = "peptideCNN3"
hyper_params3['batch_size'] = 64
hyper_params3['data_path'] = "C:\\Users\\diash\\repos\\DeepRTplus\\data\\ATLANTIS_SILICA.txt"
hyper_params3['epochs'] = 20
hyper_params3['length_peptide_sequence'] = 25
hyper_params3['number_of_filters'] = 40
hyper_params3['kernel_sizes'] = [5, 3]
hyper_params3['padding_sizes'] = [3, 2]
hyper_params3['dropout_rate'] = 0.1
hyper_params3['peptide_input_dim'] = 20
hyper_params3['learning_rate'] = 0.0001
hyper_params3['backwards'] = False


#backwards test
hyper_params4 = {}
hyper_params4['name'] = "peptideCNN4"
hyper_params4['batch_size'] = 16
hyper_params4['data_path'] = "C:\\Users\\diash\\repos\\DeepRTplus\\data\\ATLANTIS_SILICA.txt"
hyper_params4['epochs'] = 20
hyper_params4['length_peptide_sequence'] = 25
hyper_params4['number_of_filters'] = 40
hyper_params4['kernel_sizes'] = [5, 3]
hyper_params4['padding_sizes'] = [3, 2]
hyper_params4['dropout_rate'] = 0.1
hyper_params4['peptide_input_dim'] = 20
hyper_params4['learning_rate'] = 0.0001
hyper_params4['backwards'] = True

#number_of_filters
hyper_params5 = {}
hyper_params5['name'] = "peptideCNN5"
hyper_params5['batch_size'] = 16
hyper_params5['data_path'] = "C:\\Users\\diash\\repos\\DeepRTplus\\data\\ATLANTIS_SILICA.txt"
hyper_params5['epochs'] = 20
hyper_params5['length_peptide_sequence'] = 25
hyper_params5['number_of_filters'] = 20
hyper_params5['kernel_sizes'] = [5, 3]
hyper_params5['padding_sizes'] = [3, 2]
hyper_params5['dropout_rate'] = 0.1
hyper_params5['peptide_input_dim'] = 20
hyper_params5['learning_rate'] = 0.0001
hyper_params5['backwards'] = False

hyper_params6 = {}
hyper_params6['name'] = "peptideCNN6"
hyper_params6['batch_size'] = 16
hyper_params6['data_path'] = "C:\\Users\\diash\\repos\\DeepRTplus\\data\\ATLANTIS_SILICA.txt"
hyper_params6['epochs'] = 20
hyper_params6['length_peptide_sequence'] = 25
hyper_params6['number_of_filters'] = 40
hyper_params6['kernel_sizes'] = [5, 3]
hyper_params6['padding_sizes'] = [3, 2]
hyper_params6['dropout_rate'] = 0.1
hyper_params6['peptide_input_dim'] = 20
hyper_params6['learning_rate'] = 0.0001
hyper_params6['backwards'] = False

hyper_params7 = {}
hyper_params7['name'] = "peptideCNN7"
hyper_params7['batch_size'] = 16
hyper_params7['data_path'] = "C:\\Users\\diash\\repos\\DeepRTplus\\data\\ATLANTIS_SILICA.txt"
hyper_params7['epochs'] = 20
hyper_params7['length_peptide_sequence'] = 25
hyper_params7['number_of_filters'] = 60
hyper_params7['kernel_sizes'] = [5, 3]
hyper_params7['padding_sizes'] = [3, 2]
hyper_params7['dropout_rate'] = 0.1
hyper_params7['peptide_input_dim'] = 20
hyper_params7['learning_rate'] = 0.0001
hyper_params7['backwards'] = False

#kernal sizes
hyper_params8 = {}
hyper_params8['name'] = "peptideCNN8"
hyper_params8['batch_size'] = 16
hyper_params8['data_path'] = "C:\\Users\\diash\\repos\\DeepRTplus\\data\\ATLANTIS_SILICA.txt"
hyper_params8['epochs'] = 20
hyper_params8['length_peptide_sequence'] = 25
hyper_params8['number_of_filters'] = 40
hyper_params8['kernel_sizes'] = [3, 2]
hyper_params8['padding_sizes'] = [2, 1]
hyper_params8['dropout_rate'] = 0.1
hyper_params8['peptide_input_dim'] = 20
hyper_params8['learning_rate'] = 0.0001
hyper_params8['backwards'] = False

hyper_params9 = {}
hyper_params9['name'] = "peptideCNN9"
hyper_params9['batch_size'] = 16
hyper_params9['data_path'] = "C:\\Users\\diash\\repos\\DeepRTplus\\data\\ATLANTIS_SILICA.txt"
hyper_params9['epochs'] = 20
hyper_params9['length_peptide_sequence'] = 25
hyper_params9['number_of_filters'] = 40
hyper_params9['kernel_sizes'] = [7, 5]
hyper_params9['padding_sizes'] = [4, 3]
hyper_params9['dropout_rate'] = 0.1
hyper_params9['peptide_input_dim'] = 20
hyper_params9['learning_rate'] = 0.0001
hyper_params9['backwards'] = False

#drop out rate
hyper_params10 = {}
hyper_params10['name'] = "peptideCNN10"
hyper_params10['batch_size'] = 16
hyper_params10['data_path'] = "C:\\Users\\diash\\repos\\DeepRTplus\\data\\ATLANTIS_SILICA.txt"
hyper_params10['epochs'] = 20
hyper_params10['length_peptide_sequence'] = 25
hyper_params10['number_of_filters'] = 40
hyper_params10['kernel_sizes'] = [5, 3]
hyper_params10['padding_sizes'] = [3, 2]
hyper_params10['dropout_rate'] = 0.1
hyper_params10['peptide_input_dim'] = 20
hyper_params10['learning_rate'] = 0.0001
hyper_params10['backwards'] = False

hyper_params11 = {}
hyper_params11['name'] = "peptideCNN11"
hyper_params11['batch_size'] = 16
hyper_params11['data_path'] = "C:\\Users\\diash\\repos\\DeepRTplus\\data\\ATLANTIS_SILICA.txt"
hyper_params11['epochs'] = 20
hyper_params11['length_peptide_sequence'] = 25
hyper_params11['number_of_filters'] = 40
hyper_params11['kernel_sizes'] = [5, 3]
hyper_params11['padding_sizes'] = [3, 2]
hyper_params11['dropout_rate'] = 0.25
hyper_params11['peptide_input_dim'] = 20
hyper_params11['learning_rate'] = 0.0001
hyper_params11['backwards'] = False

hyper_params12 = {}
hyper_params12['name'] = "peptideCNN12"
hyper_params12['batch_size'] = 16
hyper_params12['data_path'] = "C:\\Users\\diash\\repos\\DeepRTplus\\data\\ATLANTIS_SILICA.txt"
hyper_params12['epochs'] = 20
hyper_params12['length_peptide_sequence'] = 25
hyper_params12['number_of_filters'] = 40
hyper_params12['kernel_sizes'] = [5, 3]
hyper_params12['padding_sizes'] = [3, 2]
hyper_params12['dropout_rate'] = 0.5
hyper_params12['peptide_input_dim'] = 20
hyper_params12['learning_rate'] = 0.0001
hyper_params12['backwards'] = False

#template
hyper_params = {}
hyper_params['name'] = "peptideCNN"
hyper_params['batch_size'] = 16
hyper_params['data_path'] = "C:\\Users\\diash\\repos\\DeepRTplus\\data\\ATLANTIS_SILICA.txt"
hyper_params['epochs'] = 20
hyper_params['length_peptide_sequence'] = 25
hyper_params['number_of_filters'] = 40
hyper_params['kernel_sizes'] = [5, 3]
hyper_params['padding_sizes'] = [3, 2]
hyper_params['dropout_rate'] = 0.1
hyper_params['peptide_input_dim'] = 20
hyper_params['learning_rate'] = 0.0001
hyper_params['backwards'] = False