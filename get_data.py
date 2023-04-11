import h5py

def get_data(start_idx = 0, end_idx = 699):
    h5_path = r'F:\ASCAD_data\ASCAD_databases\ASCAD.h5'
    with h5py.File(h5_path, "r") as f:
        p_data = {}
        p_data['traces'] = f['Profiling_traces']['traces'][:]
        p_data['labels'] = f['Profiling_traces']['labels'][:]
        p_data['plainx'] = f['Profiling_traces']['metadata']['plaintext']
        p_data['plainx'] = p_data['plainx'][:, 2]
        p_data['traces'] = p_data['traces'][0:45000, :]
        p_data['labels'] = p_data['labels'][0:45000]

        a_data = {}
        a_data['traces'] = f['Attack_traces']['traces'][:]
        a_data['labels'] = f['Attack_traces']['labels'][:]
        a_data['plainx'] = f['Attack_traces']['metadata']['plaintext']
        a_data['plainx'] = a_data['plainx'][:, 2]

    return p_data, a_data
