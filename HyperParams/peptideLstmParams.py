#for peptideCNN

#batch sizes
hyper_params1 = {}
hyper_params1['name'] = "peptideEnsemble1"
hyper_params1['batch_size'] = 16
hyper_params1['data_path'] = "C:\\Users\\diash\\repos\\DeepRTplus\\data\\ATLANTIS_SILICA.txt"
hyper_params1['epochs'] = 20
hyper_params1['length_peptide_sequence'] = 25
hyper_params1['number_of_filters'] = 40
hyper_params1['kernel_sizes'] = [5, 3]
hyper_params1['padding_sizes'] = [3, 2]
hyper_params1['dropout_rate'] = 0.1
hyper_params1['peptide_input_dim'] = 20
hyper_params1['learning_rate'] = 0.00001
hyper_params1['backwards'] = False

hyper_params1['hidden_size'] = 128
hyper_params1['num_layers'] = 2




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
hyper_params['learning_rate'] = 0.001
hyper_params['backwards'] = False