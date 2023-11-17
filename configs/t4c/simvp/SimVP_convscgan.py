method = 'SimVPGAN'
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'convsc'
hid_S = 32
hid_T = 128
N_T = 4
N_S = 3
lr = 1e-3
batch_size = 64
drop_path = 0.1
root_dir = "7days"
data_root = "7days"
workers = 8
epochs = 200
sched = 'onecycle'