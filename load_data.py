def data_to_dict(pids, hdf5_folder):
    """
    Store the hdf5 data to a dict
    Parameters:
        pids: list of integers
        hdf5_folder: str
    Return:
        data_dict, dictionary

    """
    data_dict = {"temperature":[], "SDNN":[], "RMSSD":[], "NN50":[], "PNN50":[], "LF":[], "HF":[], "lf_to_hf":[], "HR":[], "Labor":[]}
    
    for i, pid in enumerate(pids):
        hdf5_name = "data_dict_" + str(pid) + ".hdf5"
        hdf5_name = os.path.join(hdf5_folder,hdf5_name)
        hdf = h5py.File(hdf5_name, "r")
        
        #read data
        Temperature_data = np.array(hdf['data_dict']["temperature"]).reshape((-1,1))
        SDNN_data = np.array(hdf['data_dict']["SDNN"]).reshape((-1,1))
        RMSSD_data = np.array(hdf['data_dict']["RMSSD"]).reshape((-1,1))
        NN50_data = np.array(hdf['data_dict']["NN50"]).reshape((-1,1))
        PNN50_data = np.array(hdf['data_dict']["PNN50"]).reshape((-1,1))
        LF_data = np.array(hdf['data_dict']["LF"]).reshape((-1,1))
        HF_data = np.array(hdf['data_dict']["HF"]).reshape((-1,1))
        LF_to_HF_data = np.array(hdf['data_dict']["lf_to_hf"]).reshape((-1,1))
        HR_data = np.array(hdf['data_dict']["HR"]).reshape((-1,1))
        Labor_data = np.array(hdf['data_dict']["Labor"]).reshape((-1,1))
        
        #truncate data by date
        Labor_data = Labor_data[Labor_data >= 0]
        Temperature_data = Temperature_data[0:len(Labor_data)*288, :]
        SDNN_data = SDNN_data[0:len(Labor_data)*288, :]
        RMSSD_data = RMSSD_data[0:len(Labor_data)*288, :]
        NN50_data = NN50_data[0:len(Labor_data)*288, :]
        PNN50_data = PNN50_data[0:len(Labor_data)*288, :]
        LF_data = LF_data[0:len(Labor_data)*288, :]
        HF_data = HF_data[0:len(Labor_data)*288, :]
        LF_to_HF_data = Temperature_data[0:len(Labor_data)*288, :]
        HR_data = HR_data[0:len(Labor_data)*288, :]
            
            
        data_dict["temperature"].append(torch.from_numpy(Temperature_data).float())
        data_dict["SDNN"].append(torch.from_numpy(SDNN_data).float())
        data_dict["RMSSD"].append(torch.from_numpy(RMSSD_data).float())
        data_dict["NN50"].append(torch.from_numpy(NN50_data).float())
        data_dict["PNN50"].append(torch.from_numpy(PNN50_data).float())
        data_dict["LF"].append(torch.from_numpy(LF_data).float())
        data_dict["HF"].append(torch.from_numpy(HF_data).float())
        data_dict["lf_to_hf"].append(torch.from_numpy(LF_to_HF_data).float())
        data_dict["HR"].append(torch.from_numpy(HR_data).float())
        data_dict["Labor"].append(torch.from_numpy(Labor_data).int())
        print("%d of %d hdf5 data has been loaded" %(i + 1, len(pids)))
    return data_dict
