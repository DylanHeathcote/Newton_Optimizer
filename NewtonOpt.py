import tensorflow as tf
from tensorflow.keras.losses import MSE
from matplotlib import pyplot as plt

# The actual line
TRUE_W = 3.0
TRUE_B = 2.0

NUM_EXAMPLES = 1000

# A vector of random x values
x = tf.random.normal(shape=[NUM_EXAMPLES])

# Generate some noise
noise = tf.random.normal(shape=[NUM_EXAMPLES])

# Calculate y
y = x * TRUE_W + TRUE_B# + noise

class MyModel(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the weights to `5.0` and the bias to `0.0`
        # In practice, these should be randomly initialized
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.w * x + self.b

model = MyModel()
# Given a callable model, inputs, outputs, and a learning rate...
def train(model, x, y, f_tilde_tm1, g_tilde_tm1, h_tilde_tm1, alpha_tm1):
    with tf.GradientTape() as t2:
        with tf.GradientTape() as t:
            # Trainable variables are automatically tracked by GradientTape
            ft = tf.keras.losses.MSE(y, model(x))
        # Use GradientTape to calculate the gradients with respect to W and b
        g_t = t.gradient(ft, model.trainable_variables)
        h_t = t2.gradient(g_t,  model.trainable_variables)
        
    f_tilde_t_new = tf.add(tf.multiply(alpha, f_tilde_tm1),tf.multiply(1-alpha, ft))
    g_tilde_t_new = tf.add(tf.multiply(alpha, g_tilde_tm1),tf.multiply(1-alpha, g_t))
    
    h_tilde_t_new = tf.add(tf.multiply(alpha, h_tilde_tm1),tf.multiply(1-alpha, h_t))

    alpha_t_new = alpha*alpha_tm1+(1-alpha)
    
    u_t = tf.multiply(f_tilde_t_new,g_tilde_t_new)
    eps_t = (alpha_t_new**2)*epsilon
    
    A_t = (1/(alpha_t_new**2))*max(eps_t, tf.pow(tf.norm(g_tilde_t_new), 2), tf.tensordot(tf.norm(f_tilde_t_new), tf.norm(h_tilde_t_new), axes = 0))
    # Subtract the gradient scaled by the learning rate
    for var in model.trainable_variables:
        # Currently w is in spot 1,1 and bias is in spot 0,0
        # Need to figure out how to assign and get slots (w/o using Optimizerv2)
        model.w.assign_sub(u_t[1,1]/A_t)
        model.b.assign_sub(u_t[0,0]/A_t)
        
    return f_tilde_t_new, g_tilde_t_new, h_tilde_t_new, alpha_t_new

# Define a training loop
def training_loop(model, x, y):
    f_tilde_t = tf.zeros((2,1), tf.float32)
    g_tilde_t = tf.zeros((2,1), tf.float32)
    h_tilde_t = tf.zeros((2,1), tf.float32)
    alpha_t = 0
    
    for epoch in epochs:
        # Update the model with the single giant batch
        f_tilde_t, g_tilde_t, h_tilde_t, alpha_t = train(model, x, y, f_tilde_t, g_tilde_t, h_tilde_t, alpha_t)

        # Track this before I update
        Ws.append(model.w.numpy())
        bs.append(model.b.numpy())
        current_loss = MSE(y, model(x))

#print("Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f" %
#      (epoch, Ws[-1], bs[-1], current_loss))
model = MyModel()

# Collect the history of W-values and b-values to plot later
Ws, bs = [], []
epochs = range(10)

alpha = .1
epsilon = 10e-32

#print("Starting: W=%1.2f b=%1.2f, loss=%2.5f" %
#      (model.w, model.b, MSE(y, model(x))))

# Do the training
training_loop(model, x, y)

# Plot it
plt.plot(epochs, Ws, "r",
         epochs, bs, "b")

plt.plot([TRUE_W] * len(epochs), "r--",
         [TRUE_B] * len(epochs), "b--")

plt.legend(["W", "b", "True W", "True b"])
plt.show()
