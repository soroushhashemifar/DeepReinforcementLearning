class PlotLearning(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.mae = []
        self.val_mae = []
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.mae.append(logs.get('mae'))
        self.val_mae.append(logs.get('val_mae'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        clear_output(wait=True)
        
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="Train Loss")
        ax1.plot(self.x, self.val_losses, label="Validation loss")
        ax1.legend()
        
        ax2.plot(self.x, self.mae, label="Train MAE")
        ax2.plot(self.x, self.val_mae, label="Validation MAE")
        ax2.legend()
        
        plt.show()


class Memory:

  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = deque(maxlen=self.capacity)

  def memorize(self, data):
    self.memory.append(data)

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def size(self):
    return len(self.memory)