import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# User input
velocity = st.slider("Velocity (v/c)", 0.0, 0.99, 0.5)
time = st.number_input("Time (t)", value=1.0)

# Do Lorentz transformation
gamma = 1 / np.sqrt(1 - velocity**2)
x_prime = lambda x, t: gamma * (x - velocity * t)
ct_prime = lambda x, t: gamma * (t - velocity * x)

# Plot
fig, ax = plt.subplots()
ax.set_title("Lorentz Transformation")
ax.plot([0, x_prime(1, time)], [0, ct_prime(1, time)], label="Transformed Event")
ax.legend()
st.pyplot(fig)
