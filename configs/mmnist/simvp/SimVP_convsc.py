method = 'SimVP'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'convsc'
hid_S = 32
hid_T = 256
N_T = 4
N_S = 4
# training
lr = 1e-3
batch_size = 32
drop_path = 0
sched = 'onecycle'