import train_net

# go once
n_epochs = 10001
continue_from_ckpt = 1
restore_epoch = 250

train_net.train_net(
    n_epochs=n_epochs,
    continue_from_ckpt=continue_from_ckpt,
    restore_epoch=restore_epoch,
    reset_optimizer_params=True)
